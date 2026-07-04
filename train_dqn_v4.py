"""Phase 4 (v4): 自己対戦+相手プールによるDQN学習 (RLCard no-limit-holdem)

v3からの変更点(最強を目指すための対策):
- 相手を固定RuleBasedからプール方式に変更。エピソードごとに
  RuleBased / Random / 過去の自分(スナップショット) からランダムに選ぶ
  → 固定相手への過剰適合を防ぎ、汎用的に強くする
- --snapshot_every(既定5000)ごとに現在の自分をプールに追加(最大 --pool_size 体)
- 評価は RuleBased と Random の両方に対して行い、平均スコアでベストを判定
- --resume で v2/v3 のモデルから継続可能

実行例:
    python train_dqn_v4.py --resume experiments/dqn_v3/dqn_model_best.pth --num_episodes 100000
"""

import argparse
import copy
import os
import random
import time

import torch

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import (
    Logger,
    plot_curve,
    reorganize,
    set_seed,
    tournament,
)

from my_agents import RuleBasedAgent


def fmt_sec(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class FrozenAgent:
    """過去の自分のスナップショット。学習せず、常にグリーディに打つ"""

    use_raw = False

    def __init__(self, agent):
        self._agent = agent

    def step(self, state):
        action, _ = self._agent.eval_step(state)
        return action

    def eval_step(self, state):
        return self._agent.eval_step(state)


def snapshot(agent):
    """リプレイメモリを除いた軽量コピーを作る"""
    memory = agent.memory
    agent.memory = None
    frozen = FrozenAgent(copy.deepcopy(agent))
    agent.memory = memory
    return frozen


def load_agent(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 古いtorchはweights_only引数なし
        return torch.load(path, map_location=device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    env = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    eval_env_rule = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    eval_env_rand = rlcard.make("no-limit-holdem", config={"seed": args.seed + 1})

    if args.resume:
        print(f"モデルを読み込んで再開: {args.resume}", flush=True)
        agent = load_agent(args.resume, device)
    else:
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[128, 128],
            replay_memory_size=100000,
            update_target_estimator_every=1000,
            epsilon_decay_steps=args.num_episodes,
            batch_size=64,
            learning_rate=5e-5,
            train_every=1,
            device=device,
        )

    rule_op = RuleBasedAgent(num_actions=env.num_actions)
    rand_op = RandomAgent(num_actions=env.num_actions)
    pool = []  # 過去の自分(FrozenAgent)

    eval_env_rule.set_agents([agent, rule_op])
    eval_env_rand.set_agents([agent, rand_op])

    def pick_opponent():
        r = random.random()
        if pool and r < args.selfplay_ratio:
            return random.choice(pool)
        if r < args.selfplay_ratio + 0.1:
            return rand_op
        return rule_op

    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.log_dir, "dqn_model_best.pth")
    last_path = os.path.join(args.log_dir, "dqn_model_last.pth")
    best_score = float("-inf")
    history = []  # (episode, vs_rule, vs_rand, score, is_best)
    start = time.time()

    print(f"学習開始: {args.num_episodes}ハンド, 自己対戦比率 {args.selfplay_ratio}, "
          f"スナップショット {args.snapshot_every}ごと(最大{args.pool_size}体), "
          f"device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            env.set_agents([agent, pick_opponent()])
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                agent.feed(ts)

            if episode % args.snapshot_every == 0:
                pool.append(snapshot(agent))
                if len(pool) > args.pool_size:
                    pool.pop(0)  # 古いものから捨てる

            if episode % args.evaluate_every == 0:
                vs_rule = tournament(eval_env_rule, args.num_eval_games)[0]
                vs_rand = tournament(eval_env_rand, args.num_eval_games)[0]
                score = (vs_rule + vs_rand) / 2
                logger.log_performance(episode, score)

                is_best = score > best_score
                if is_best:
                    best_score = score
                    torch.save(agent, best_path)
                history.append((episode, vs_rule, vs_rand, score, is_best))

                elapsed = time.time() - start
                eta = elapsed / episode * (args.num_episodes - episode)
                print(f"[{episode:>6}/{args.num_episodes}] "
                      f"vsRule {vs_rule:+.3f} vsRand {vs_rand:+.3f} "
                      f"平均 {score:+.3f} BB/hand "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

            if episode % args.checkpoint_every == 0:
                torch.save(agent, last_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    torch.save(agent, last_path)
    plot_curve(csv_path, fig_path, "DQN self-play (v4)")

    # ---- 評価成績の推移一覧 ----
    print("\n===== 評価成績の推移 (BB/hand) =====")
    print(f"{'ハンド':>8} | {'vs Rule':>9} | {'vs Random':>9} | {'平均':>9} | ベスト")
    print("-" * 60)
    for ep, r1, r2, sc, b in history:
        print(f"{ep:>8} | {r1:>+9.3f} | {r2:>+9.3f} | {sc:>+9.3f} | {'★' if b else ''}")
    print("-" * 60)
    print(f"ベスト平均スコア: {best_score:+.4f} BB/hand ({best_score * 100:+.1f} BB/100)")
    print(f"ベストモデル: {best_path}")
    print(f"最終モデル(再開用): {last_path}")
    print(f"学習曲線: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DQN training v4 (self-play pool)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--evaluate_every", type=int, default=2500)
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--snapshot_every", type=int, default=5000)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--selfplay_ratio", type=float, default=0.5,
                        help="過去の自分と対戦する確率(残りはRandom 0.1 / RuleBased)")
    parser.add_argument("--resume", type=str, default=None,
                        help="継続する既存モデル(.pth)のパス")
    parser.add_argument("--log_dir", type=str, default="experiments/dqn_v4")
    args = parser.parse_args()
    train(args)
