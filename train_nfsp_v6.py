"""エンジンv6: equity観測付きNFSP自己対戦 (ポストフロップ特化)

構成:
- 環境: extended_env_v6 (70次元: v5の68+equity+SPR)
- 2体のNFSPが自己対戦。プリフロップは全員GTO表で固定し、
  ポストフロップの判断だけを学習する(v5と同じ思想)
- 評価は GTO化RuleBased 相手
- --until 対応(例: --until 07:00 で朝まで学習)

実行例:
    python train_nfsp_v6.py --num_episodes 30000          # 試運転
    python train_nfsp_v6.py --num_episodes 10000000 --until 07:00
    python train_nfsp_v6.py --resume experiments/nfsp_v6/nfsp_last.pth --num_episodes 10000000 --until 07:00
"""

import argparse
import datetime
import os
import time

import torch

from rlcard.agents import NFSPAgent
from rlcard.utils import Logger, plot_curve, reorganize, set_seed, tournament

from extended_env import PreflopGTO, is_postflop_transition
from extended_env_v6 import make_extended_env_v6
from my_agents import RuleBasedAgent


def fmt_sec(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    env = make_extended_env_v6(num_players=2, seed=args.seed)
    eval_env = make_extended_env_v6(num_players=2, seed=args.seed + 1)

    if args.resume:
        print(f"再開: {args.resume}", flush=True)
        agents = load_torch(args.resume, device)
    else:
        agents = [
            NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[256, 256],
                q_mlp_layers=[256, 256],
                q_replay_memory_size=200000,
                q_epsilon_decay_steps=args.epsilon_decay_steps,
                device=device,
            )
            for _ in range(2)
        ]

    wrapped = [PreflopGTO(a) for a in agents]  # プリフロップはGTO表
    env.set_agents(wrapped)

    rule_op = PreflopGTO(RuleBasedAgent(num_actions=env.num_actions))
    eval_env.set_agents([wrapped[0], rule_op])

    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.log_dir, "nfsp_model_best.pth")  # agents[0](デプロイ用)
    last_path = os.path.join(args.log_dir, "nfsp_last.pth")        # 2体セット(再開用)
    best_reward = float("-inf")
    history = []
    start = time.time()

    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        now = datetime.datetime.now()
        dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if dt <= now:
            dt += datetime.timedelta(days=1)
        deadline = dt.timestamp()
        print(f"終了予定時刻: {dt:%m/%d %H:%M}", flush=True)

    print(f"v6 NFSP学習開始: {args.num_episodes}ハンド (70次元観測, "
          f"ポストフロップ特化), device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            if deadline and time.time() >= deadline:
                print(f"指定時刻に到達。{episode - 1}ハンドで終了します", flush=True)
                break

            for a in agents:
                if hasattr(a, "sample_episode_policy"):
                    a.sample_episode_policy()

            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for i in range(2):
                for ts in trajectories[i]:
                    if is_postflop_transition(ts):
                        agents[i].feed(ts)

            if episode % args.evaluate_every == 0:
                reward = tournament(eval_env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)

                is_best = reward > best_reward
                if is_best:
                    best_reward = reward
                    torch.save(agents[0], best_path)
                history.append((episode, reward, is_best))

                elapsed = time.time() - start
                eta = elapsed / episode * (args.num_episodes - episode)
                print(f"[{episode:>7}/{args.num_episodes}] "
                      f"vs GTO-Rule {reward:+.4f} BB/hand "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

            if episode % args.checkpoint_every == 0:
                torch.save(agents, last_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    torch.save(agents, last_path)
    plot_curve(csv_path, fig_path, "NFSP v6 (equity obs)")

    print("\n===== 評価成績の推移 (BB/hand, vs GTO化RuleBased) =====")
    print(f"{'ハンド':>9} | {'収支':>9} | ベスト")
    print("-" * 34)
    for ep, r, b in history:
        print(f"{ep:>9} | {r:>+9.3f} | {'★' if b else ''}")
    print("-" * 34)
    print(f"ベスト: {best_reward:+.4f} BB/hand")
    print(f"ベストモデル(デプロイ用): {best_path}")
    print(f"再開用: {last_path}")
    print("※このモデルは70次元観測用。デプロイには extended_env_v6 が必要")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NFSP v6 training (equity observation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=300000)
    parser.add_argument("--evaluate_every", type=int, default=5000)
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--epsilon_decay_steps", type=int, default=300000)
    parser.add_argument("--until", type=str, default=None,
                        help="この時刻で保存して終了 (例: 07:00)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="experiments/nfsp_v6")
    args = parser.parse_args()
    train(args)
