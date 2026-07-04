"""v5(DQN, 68次元) vs v6(NFSP, 70次元) の直接対戦

v6環境(70次元)で対戦し、v5には先頭68次元だけ見せるアダプタを噛ませる
(v6観測はv5観測の末尾にequity/SPRを足しただけなので互換)。
両者ともプリフロップはGTO表(本番デプロイと同条件)。

実行例:
    python compare_v5_v6.py
    python compare_v5_v6.py --num_games 20000
"""

import argparse

import torch

from rlcard.utils import set_seed, tournament

from extended_env import PreflopGTO
from extended_env_v6 import make_extended_env_v6

V5_DIM = 68


class ObsSlice:
    """観測の先頭dim次元だけをエージェントに見せるアダプタ"""
    use_raw = False

    def __init__(self, agent, dim=V5_DIM):
        self.agent = agent
        self.dim = dim

    def _cut(self, state):
        s = dict(state)
        s["obs"] = state["obs"][:self.dim]
        return s

    def step(self, state):
        return self.agent.step(self._cut(state))

    def eval_step(self, state):
        return self.agent.eval_step(self._cut(state))


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("v5 vs v6 head-to-head")
    parser.add_argument("--v5", default="experiments/dqn_v5/dqn_model_best.pth")
    parser.add_argument("--v6", default="experiments/nfsp_v6/nfsp_model_best.pth")
    parser.add_argument("--num_games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    v5 = PreflopGTO(ObsSlice(load(args.v5)))
    v6 = PreflopGTO(load(args.v6))
    env = make_extended_env_v6(num_players=2, seed=args.seed)

    print(f"v6: {args.v6}\nv5: {args.v5}")
    print(f"各席 {args.num_games} ハンド(計 {args.num_games * 2})で対戦中...\n")

    env.set_agents([v6, v5])
    a = tournament(env, args.num_games)[0]
    print(f"v6が席0のとき: v6 {a:+.4f} BB/hand")

    env.set_agents([v5, v6])
    b = tournament(env, args.num_games)[1]
    print(f"v6が席1のとき: v6 {b:+.4f} BB/hand")

    avg = (a + b) / 2
    print(f"\nv6の平均収支: {avg:+.4f} BB/hand ({avg * 100:+.1f} BB/100)")
    print("→ " + ("v6 の方が強い" if avg > 0 else "v5 の方が強い" if avg < 0 else "互角"))
