r"""v7(NFSP) の強さを、今の上級モデル(v5 / v4)と直接対戦で測る。

観測次元が違うので、全員を v6環境(70次元)で戦わせ、
v5(68)/v4(54) には obs の先頭だけを見せる ObsSlice で合わせる
(compare_v5_v6.py と同じ考え方。v7 は v6 と同じ 70次元env)。

プリフロップは全員 GTO 固定(PreflopGTO)にして、
ポストフロップの地力だけを比較する ＝ 実デプロイ(GTOハイブリッド)に近い条件。
席を入れ替えて BB/hand を測るので、席順の有利不利は相殺される。

使い方(ローカル / venv):
    .\venv\Scripts\python.exe compare_v7.py
    .\venv\Scripts\python.exe compare_v7.py --num_games 20000        # ブレを減らす
    .\venv\Scripts\python.exe compare_v7.py --v5 models\heads_up_advanced.pth  # 本番slimと比較
    .\venv\Scripts\python.exe compare_v7.py --with_v4               # v4とも対戦
    .\venv\Scripts\python.exe compare_v7.py --raw                   # プリフロップGTOを外して素の地力

判定: +BB/hand が v7 のプラス。おおむね +0.3 以上で「明確に勝ち」、
      ±0.3 以内は「誤差(=載せ替えの意味が薄い)」の目安。
"""

import argparse
import os

import numpy as np
import torch

from rlcard.utils import set_seed, tournament

from extended_env_v6 import make_extended_env_v6
from extended_env import PreflopGTO


def load(path, device="cpu"):
    if not os.path.exists(path):
        raise SystemExit(f"見つからない: {path}")
    try:
        m = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        m = torch.load(path, map_location=device)
    if isinstance(m, (list, tuple)):  # nfsp_last 形式(list)なら先頭
        m = m[0]
    return m


class ObsSlice:
    """内側エージェントに obs の先頭 dim 次元だけ見せるアダプタ。

    v5(68)/v4(54) を 70次元環境で戦わせるために使う。
    legal_actions や raw_obs はそのまま(プリフロップGTOはこれを見る)。
    """

    def __init__(self, agent, dim):
        self.agent = agent
        self.dim = dim
        self.use_raw = False

    def _cut(self, state):
        obs = state.get("obs")
        if obs is None:
            return state
        obs = np.asarray(obs)
        if obs.shape[-1] <= self.dim:
            return state
        s = dict(state)
        s["obs"] = obs[..., : self.dim]
        return s

    def eval_step(self, state):
        return self.agent.eval_step(self._cut(state))

    def step(self, state):
        if hasattr(self.agent, "step"):
            return self.agent.step(self._cut(state))
        return self.eval_step(state)[0]


def wrap(model, dim, preflop_gto):
    """モデルを次元合わせし、必要ならプリフロップGTOで包む。"""
    inner = ObsSlice(model, dim)
    return PreflopGTO(inner) if preflop_gto else inner


def h2h(env, a, b, num_games):
    """a を席入替で b と対戦させ、a の平均 BB/hand を返す(正 = a の勝ち)。"""
    env.set_agents([a, b])
    a_as_p0 = tournament(env, num_games)[0]
    env.set_agents([b, a])
    a_as_p1 = tournament(env, num_games)[1]
    return (a_as_p0 + a_as_p1) / 2.0


def verdict(bb):
    if bb > 0.3:
        return "→ v7 の明確な勝ち。載せ替えの価値あり"
    if bb < -0.3:
        return "→ v7 の負け。載せ替えは劣化。見送り推奨"
    return "→ 誤差レベル。載せ替えの意味は薄い(現状維持でよい)"


def main():
    p = argparse.ArgumentParser("compare v7 vs v5/v4")
    p.add_argument("--v7", default=os.path.join("experiments", "nfsp_v7", "nfsp_model_best.pth"))
    p.add_argument("--v5", default=os.path.join("experiments", "dqn_v5", "dqn_model_best.pth"))
    p.add_argument("--v4", default=os.path.join("experiments", "dqn_v4", "dqn_model_best.pth"))
    p.add_argument("--with_v4", action="store_true", help="v4 とも対戦する")
    p.add_argument("--num_games", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="プリフロップGTOを外して素の地力を測る")
    args = p.parse_args()

    preflop_gto = not args.raw
    set_seed(args.seed)
    env = make_extended_env_v6(num_players=2, seed=args.seed)

    print(f"環境: extended_env_v6 (70次元) / {args.num_games}ハンド x 席入替 "
          f"/ プリフロップGTO={'ON' if preflop_gto else 'OFF'}\n", flush=True)

    v7 = wrap(load(args.v7), 70, preflop_gto)

    # v7 vs v5
    v5 = wrap(load(args.v5), 68, preflop_gto)
    bb5 = h2h(env, v7, v5, args.num_games)
    print(f"v7 vs v5 : {bb5:+.3f} BB/hand   {verdict(bb5)}", flush=True)

    # v7 vs v4 (任意)
    if args.with_v4:
        v4 = wrap(load(args.v4), 54, preflop_gto)
        bb4 = h2h(env, v7, v4, args.num_games)
        print(f"v7 vs v4 : {bb4:+.3f} BB/hand   {verdict(bb4)}", flush=True)

    print("\n※ num_games が小さいとブレる。判断に迷う値なら --num_games 20000 で再測定を。")


if __name__ == "__main__":
    main()
