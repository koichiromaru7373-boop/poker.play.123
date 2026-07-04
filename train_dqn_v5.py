"""v5: 拡張観測+プリフロップGTO固定のポストフロップ特化DQN学習

v4からの変更点:
- 拡張観測環境(extended_env.py, 68次元)を使用
- 全員(自分も相手も)プリフロップはGTO表で打つ
  → 学習はポストフロップの判断のみに集中(プリフロップ遷移はリプレイに入れない)
- 相手プール: GTO化RuleBased / GTO化Random / 過去の自分
- 評価はGTO化RuleBased相手(1000ハンド)

※旧モデル(v2〜v4)とは観測次元が違うため resume・直接対戦は不可。新系統。

実行例:
    python train_dqn_v5.py --num_episodes 30000   # 試運転(15〜20分)
    python train_dqn_v5.py                        # 本番(既定300000, 数時間)
    python train_dqn_v5.py --resume experiments/dqn_v5/dqn_model_last.pth
"""

import argparse
import copy
import datetime
import os
import random
import time

import torch

from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import Logger, plot_curve, reorganize, set_seed, tournament

from extended_env import PreflopGTO, is_postflop_transition, make_extended_env
from my_agents import RuleBasedAgent


def fmt_sec(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def snapshot(agent):
    memory = agent.memory
    agent.memory = None
    frozen = PreflopGTO(copy.deepcopy(agent))
    agent.memory = memory
    return frozen


def load_agent(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    env = make_extended_env(num_players=2, seed=args.seed)
    eval_env = make_extended_env(num_players=2, seed=args.seed + 1)

    if args.resume:
        print(f"再開: {args.resume}", flush=True)
        agent = load_agent(args.resume, device)
    else:
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[256, 256],
            replay_memory_size=200000,
            update_target_estimator_every=1000,
            epsilon_decay_steps=args.epsilon_decay_steps,
            batch_size=64,
            learning_rate=5e-5,
            train_every=1,
            device=device,
        )

    learner = PreflopGTO(agent)  # 自分もプリフロップはGTO表
    rule_op = PreflopGTO(RuleBasedAgent(num_actions=env.num_actions))
    rand_op = PreflopGTO(RandomAgent(num_actions=env.num_actions))
    pool = []

    eval_env.set_agents([learner, rule_op])

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
    best_reward = float("-inf")
    history = []
    start = time.time()

    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        now = datetime.datetime.now()
        dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if dt <= now:
            dt += datetime.timedelta(days=1)  # 過ぎていたら翌日のその時刻
        deadline = dt.timestamp()
        print(f"終了予定時刻: {dt:%m/%d %H:%M}", flush=True)

    print(f"v5学習開始: {args.num_episodes}ハンド(ポストフロップ特化, 68次元観測), "
          f"device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            if deadline and time.time() >= deadline:
                print(f"指定時刻に到達。{episode - 1}ハンドで終了します", flush=True)
                break
            env.set_agents([learner, pick_opponent()])
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                if is_postflop_transition(ts):  # ポストフロップの判断だけ学習
                    agent.feed(ts)

            if episode % args.snapshot_every == 0:
                pool.append(snapshot(agent))
                if len(pool) > args.pool_size:
                    pool.pop(0)

            if episode % args.evaluate_every == 0:
                reward = tournament(eval_env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)

                is_best = reward > best_reward
                if is_best:
                    best_reward = reward
                    torch.save(agent, best_path)
                history.append((episode, reward, is_best))

                elapsed = time.time() - start
                eta = elapsed / episode * (args.num_episodes - episode)
                print(f"[{episode:>6}/{args.num_episodes}] "
                      f"vs GTO-Rule {reward:+.4f} BB/hand "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

            if episode % args.checkpoint_every == 0:
                torch.save(agent, last_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    torch.save(agent, last_path)
    plot_curve(csv_path, fig_path, "DQN v5 (postflop focus)")

    print("\n===== 評価成績の推移 (BB/hand, vs GTO化RuleBased) =====")
    print(f"{'ハンド':>8} | {'収支':>9} | ベスト")
    print("-" * 32)
    for ep, r, b in history:
        print(f"{ep:>8} | {r:>+9.3f} | {'★' if b else ''}")
    print("-" * 32)
    print(f"ベスト: {best_reward:+.4f} BB/hand")
    print(f"ベストモデル: {best_path}")
    print(f"再開用: {last_path}")
    print("※このモデルは拡張観測(68次元)用。デプロイには extended_env が必要")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DQN v5: postflop-focused training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=300000)
    parser.add_argument("--evaluate_every", type=int, default=2500)
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--snapshot_every", type=int, default=5000)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--selfplay_ratio", type=float, default=0.5)
    parser.add_argument("--epsilon_decay_steps", type=int, default=300000)
    parser.add_argument("--until", type=str, default=None,
                        help="この時刻に到達したら保存して終了 (例: 17:00)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="experiments/dqn_v5")
    args = parser.parse_args()
    train(args)
