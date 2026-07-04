"""Phase 3 (v2): DQN学習スクリプト (RLCard no-limit-holdem)

v1からの変更点:
- 対戦相手は RuleBasedAgent(テスト相手と一致)
- 学習 50000 ハンド(エピソード)
- 2500 ハンドごとに 500 ハンド対戦で評価
- 評価成績がベスト更新のたびに dqn_model_best.pth に保存
- 途中経過(経過時間・残り時間の目安)を評価ごとに表示
- 最後に評価成績の推移を一覧表示

実行例:
    python train_dqn_v2.py
"""

import argparse
import os
import time

import torch

import rlcard
from rlcard.agents import DQNAgent
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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    env = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    eval_env = rlcard.make("no-limit-holdem", config={"seed": args.seed})

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
    opponent = RuleBasedAgent(num_actions=env.num_actions)
    env.set_agents([agent, opponent])
    eval_env.set_agents([agent, opponent])

    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.log_dir, "dqn_model_best.pth")
    best_reward = float("-inf")
    history = []  # (episode, reward, is_best)
    start = time.time()

    print(f"学習開始: {args.num_episodes}ハンド, 評価: {args.evaluate_every}ごとに"
          f"{args.num_eval_games}ハンド, device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                agent.feed(ts)

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
                      f"平均収支 {reward:+.4f} BB/hand ({reward * 100:+.1f} BB/100) "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    plot_curve(csv_path, fig_path, "DQN vs RuleBased (v2)")

    # ---- 評価成績の推移一覧 ----
    print("\n===== 評価成績の推移 =====")
    print(f"{'ハンド':>8} | {'平均収支/hand':>14} | {'BB/100':>9} | ベスト")
    print("-" * 45)
    for ep, r, b in history:
        print(f"{ep:>8} | {r:>+14.4f} | {r * 100:>+9.1f} | {'★' if b else ''}")
    print("-" * 45)
    print(f"ベスト成績: {best_reward:+.4f} BB/hand ({best_reward * 100:+.1f} BB/100)")
    print(f"ベストモデル: {best_path}")
    print(f"学習曲線: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DQN training v2 on no-limit-holdem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=50000)
    parser.add_argument("--evaluate_every", type=int, default=2500)
    parser.add_argument("--num_eval_games", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="experiments/dqn_v2")
    args = parser.parse_args()
    train(args)
