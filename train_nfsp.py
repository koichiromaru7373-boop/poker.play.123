"""Phase 4 (NFSP): 自己対戦で均衡戦略に近づける学習 (RLCard no-limit-holdem)

DQNが頭打ちになったため手法を変更。NFSPは2体の自己対戦で
「搾取されにくいバランス型戦略」に収束していく、ポーカー向けの正攻法。

- 2体のNFSPAgentが自己対戦し、両方が学習する
- 評価は agents[0] の平均方針(average policy)で行う
- vs RuleBased に加え、--benchmark のDQNモデル(既定: dqn_v4ベスト)とも対戦させ、
  「今のチャンピオンに勝てるか」を直接測る
- ベスト判定は vs benchmark の成績(なければ vs Rule)

実行例:
    python train_nfsp.py
    python train_nfsp.py --resume experiments/nfsp/nfsp_last.pth --num_episodes 100000
"""

import argparse
import os
import time

import torch

import rlcard
from rlcard.agents import NFSPAgent
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


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 古いtorchはweights_only引数なし
        return torch.load(path, map_location=device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    env = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    eval_env_rule = rlcard.make("no-limit-holdem", config={"seed": args.seed})
    eval_env_bench = rlcard.make("no-limit-holdem", config={"seed": args.seed + 1})

    if args.resume:
        print(f"再開: {args.resume}", flush=True)
        agents = load_torch(args.resume, device)
    else:
        agents = [
            NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[128, 128],
                q_mlp_layers=[128, 128],
                q_replay_memory_size=100000,
                q_epsilon_decay_steps=args.num_episodes,
                device=device,
            )
            for _ in range(2)
        ]
    env.set_agents(agents)

    rule_op = RuleBasedAgent(num_actions=env.num_actions)
    eval_env_rule.set_agents([agents[0], rule_op])

    bench_agent = None
    if args.benchmark and os.path.exists(args.benchmark):
        print(f"ベンチマーク相手: {args.benchmark}", flush=True)
        bench_agent = load_torch(args.benchmark, device)
        eval_env_bench.set_agents([agents[0], bench_agent])
    else:
        print("ベンチマークモデルなし(vs Ruleでベスト判定)", flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.log_dir, "nfsp_model_best.pth")   # agents[0]のみ(デプロイ用)
    last_path = os.path.join(args.log_dir, "nfsp_last.pth")         # 2体セット(再開用)
    best_score = float("-inf")
    history = []  # (episode, vs_rule, vs_bench, is_best)
    start = time.time()

    print(f"NFSP学習開始: {args.num_episodes}ハンド, device={device}", flush=True)

    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            for a in agents:
                if hasattr(a, "sample_episode_policy"):
                    a.sample_episode_policy()

            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for i in range(2):
                for ts in trajectories[i]:
                    agents[i].feed(ts)

            if episode % args.evaluate_every == 0:
                vs_rule = tournament(eval_env_rule, args.num_eval_games)[0]
                vs_bench = None
                if bench_agent is not None:
                    vs_bench = tournament(eval_env_bench, args.num_eval_games)[0]
                score = vs_bench if vs_bench is not None else vs_rule
                logger.log_performance(episode, score)

                is_best = score > best_score
                if is_best:
                    best_score = score
                    torch.save(agents[0], best_path)
                history.append((episode, vs_rule, vs_bench, is_best))

                elapsed = time.time() - start
                eta = elapsed / episode * (args.num_episodes - episode)
                bench_str = f" vsDQN {vs_bench:+.3f}" if vs_bench is not None else ""
                print(f"[{episode:>6}/{args.num_episodes}] "
                      f"vsRule {vs_rule:+.3f}{bench_str} BB/hand "
                      f"経過 {fmt_sec(elapsed)} 残り目安 {fmt_sec(eta)}"
                      f"{'  ** ベスト更新 → 保存 **' if is_best else ''}",
                      flush=True)

            if episode % args.checkpoint_every == 0:
                torch.save(agents, last_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    torch.save(agents, last_path)
    plot_curve(csv_path, fig_path, "NFSP self-play")

    print("\n===== 評価成績の推移 (BB/hand) =====")
    print(f"{'ハンド':>8} | {'vs Rule':>9} | {'vs DQN':>9} | ベスト")
    print("-" * 45)
    for ep, r, b, is_b in history:
        b_str = f"{b:>+9.3f}" if b is not None else "     ----"
        print(f"{ep:>8} | {r:>+9.3f} | {b_str} | {'★' if is_b else ''}")
    print("-" * 45)
    print(f"ベストスコア: {best_score:+.4f} BB/hand ({best_score * 100:+.1f} BB/100)")
    print(f"ベストモデル(デプロイ用): {best_path}")
    print(f"再開用: {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NFSP self-play training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=200000)
    parser.add_argument("--evaluate_every", type=int, default=5000)
    parser.add_argument("--num_eval_games", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10000)
    parser.add_argument("--benchmark", type=str,
                        default=os.path.join("experiments", "dqn_v4", "dqn_model_best.pth"),
                        help="対戦評価に使うDQNモデル(.pth)")
    parser.add_argument("--resume", type=str, default=None, help="nfsp_last.pth のパス")
    parser.add_argument("--log_dir", type=str, default="experiments/nfsp")
    args = parser.parse_args()
    train(args)
