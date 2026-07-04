"""Phase 4.5: 6人テーブル用DQN学習 (RLCard no-limit-holdem, 6-max)

自社サイトの「利用者+AI5人」用モデル。ヘッズアップとは最適戦略が
大きく異なるため、6人テーブルで学習し直す。

- 席0がDQN、席1〜5は RuleBased / Random / 過去の自分 の混合(v4と同じ発想)
- 評価は vs RuleBased×5 で実施
- --resume 対応(ヘッズアップ用モデルは構造同一なので流用可能だが、ゼロから推奨)

実行例:
    python train_dqn_6max.py --num_episodes 100000
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

NUM_PLAYERS = 6


def fmt_sec(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class FrozenAgent:
    use_raw = False

    def __init__(self, agent):
        self._agent = agent

    def step(self, state):
        action, _ = self._agent.eval_step(state)
        return action

    def eval_step(self, state):
        return self._agent.eval_step(state)


def snapshot(agent):
    memory = agent.memory
    agent.memory = None
    frozen = FrozenAgent(copy.deepcopy(agent))
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

    config = {"seed": args.seed, "game_num_players": NUM_PLAYERS}
    env = rlcard.make("no-limit-holdem", config=config)
    eval_env = rlcard.make("no-limit-holdem", config=config)

    if args.resume:
        print(f"再開: {args.resume}", flush=True)
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
    pool = []

    eval_env.set_agents([agent] + [RuleBasedAgent(num_actions=env.num_actions)
                                   for _ in range(NUM_PLAYERS - 1)])

    def pick_opponent():
        r = random.random()
        if pool and r < args.selfplay_ratio:
            return random.choice(pool)
        if r < args.selfplay_ratio + 0.1:
            return rand_op
        return rule_op

    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.log_dir, "dqn6_model_best.pth")
    last_path = os.path.join(args.log_dir, "dqn6_model_last.pth")
    best_reward = float("-inf")
    history = []
    start = time.time()

    print(f"6-max学習開始: {args.num_episodes}ハンド, device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            env.set_agents([agent] + [pick_opponent() for _ in range(NUM_PLAYERS - 1)])
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
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
                      f"vs Rule×5 {reward:+.4f} BB/hand "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

            if episode % args.checkpoint_every == 0:
                torch.save(agent, last_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    torch.save(agent, last_path)
    plot_curve(csv_path, fig_path, "DQN 6-max")

    print("\n===== 評価成績の推移 (BB/hand, vs Rule×5) =====")
    print(f"{'ハンド':>8} | {'収支':>9} | ベスト")
    print("-" * 32)
    for ep, r, b in history:
        print(f"{ep:>8} | {r:>+9.3f} | {'★' if b else ''}")
    print("-" * 32)
    print(f"ベスト: {best_reward:+.4f} BB/hand")
    print(f"ベストモデル: {best_path}")
    print(f"再開用: {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DQN training for 6-max table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--evaluate_every", type=int, default=2500)
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--snapshot_every", type=int, default=5000)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--selfplay_ratio", type=float, default=0.4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="experiments/dqn_6max")
    args = parser.parse_args()
    train(args)
