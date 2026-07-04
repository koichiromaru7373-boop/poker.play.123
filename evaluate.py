"""Phase 4: 保存済みDQNモデルの再評価スクリプト

500ハンド評価はブレが大きいため、大きなサンプルで測り直す。
実行例:
    python evaluate.py experiments\\dqn_v2\\dqn_model_best.pth
    python evaluate.py experiments\\dqn_v2\\dqn_model_best.pth experiments\\dqn_v3\\dqn_model_best.pth --num_games 20000
"""

import argparse

import torch

import rlcard
from rlcard.utils import set_seed, tournament

from my_agents import RuleBasedAgent


def load_agent(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 古いtorchはweights_only引数なし
        return torch.load(path, map_location=device)


def evaluate(model_path, num_games, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = rlcard.make("no-limit-holdem", config={"seed": seed})
    agent = load_agent(model_path, device)
    env.set_agents([agent, RuleBasedAgent(num_actions=env.num_actions)])
    return tournament(env, num_games)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate saved DQN models vs RuleBasedAgent")
    parser.add_argument("models", nargs="+", help="評価するモデル(.pth)のパス(複数可)")
    parser.add_argument("--num_games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"各モデルを {args.num_games} ハンドで評価 (vs RuleBasedAgent)\n")
    results = []
    for path in args.models:
        r = evaluate(path, args.num_games, args.seed)
        results.append((path, r))
        print(f"{path}: {r:+.4f} BB/hand ({r * 100:+.1f} BB/100)")

    if len(results) > 1:
        best = max(results, key=lambda x: x[1])
        print(f"\n最良モデル: {best[0]} ({best[1] * 100:+.1f} BB/100)")
