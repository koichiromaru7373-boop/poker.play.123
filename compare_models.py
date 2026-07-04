"""モデル同士の直接対戦(ヘッズアップ)で強さを比較する

席順の有利不利を消すため、席を入れ替えて2回対戦し平均を取る。
実行例:
    python compare_models.py experiments\\dqn_v4\\dqn_model_best.pth experiments\\dqn_v3\\dqn_model_best.pth
    python compare_models.py experiments\\dqn_v4\\dqn_model_last.pth experiments\\dqn_v4\\dqn_model_best.pth --num_games 20000
"""

import argparse

import torch

import rlcard
from rlcard.utils import set_seed, tournament


def load_agent(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 古いtorchはweights_only引数なし
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Head-to-head model comparison")
    parser.add_argument("model_a", help="モデルA(.pth)")
    parser.add_argument("model_b", help="モデルB(.pth)")
    parser.add_argument("--num_games", type=int, default=10000,
                        help="片側あたりの対戦ハンド数(合計はこの2倍)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    agent_a = load_agent(args.model_a, device)
    agent_b = load_agent(args.model_b, device)

    env = rlcard.make("no-limit-holdem", config={"seed": args.seed})

    print(f"A: {args.model_a}")
    print(f"B: {args.model_b}")
    print(f"各席 {args.num_games} ハンド(計 {args.num_games * 2} ハンド)で対戦中...\n")

    env.set_agents([agent_a, agent_b])
    a_seat0 = tournament(env, args.num_games)[0]
    print(f"Aが席0のとき: A {a_seat0:+.4f} BB/hand")

    env.set_agents([agent_b, agent_a])
    a_seat1 = tournament(env, args.num_games)[1]
    print(f"Aが席1のとき: A {a_seat1:+.4f} BB/hand")

    avg = (a_seat0 + a_seat1) / 2
    print(f"\nAの平均収支: {avg:+.4f} BB/hand ({avg * 100:+.1f} BB/100)")
    if avg > 0:
        print("→ A の方が強い")
    elif avg < 0:
        print("→ B の方が強い")
    else:
        print("→ 互角")
