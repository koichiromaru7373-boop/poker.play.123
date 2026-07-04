"""v6: equity(勝率)とSPRを観測に加えた拡張環境 (70次元)

v5の68次元(標準54+14)に以下2次元を追加:
- equity: 現時点の勝率(モンテカルロ60試行、キャッシュ付き)
- SPR: スタック/ポット比(20でクリップして正規化)

※v5モデル(68次元)ともv4以前(54次元)とも互換なし。新系統。
※equity計算の分だけ学習は遅くなる(体感1.5〜2倍)。
"""

import numpy as np

from equity import estimate_win_prob
from extended_env import NUM_EXTRA as NUM_EXTRA_V5, ExtendedNLHEnv

NUM_EXTRA_V6 = NUM_EXTRA_V5 + 2
EQ_SAMPLES = 60

_eq_cache = {}


def cached_equity(hand, public, num_opponents):
    key = (tuple(sorted(hand)), tuple(public), num_opponents)
    v = _eq_cache.get(key)
    if v is None:
        v = estimate_win_prob(hand, public, num_opponents,
                              num_samples=EQ_SAMPLES)
        if len(_eq_cache) > 300000:  # メモリ保護
            _eq_cache.clear()
        _eq_cache[key] = v
    return v


def make_extended_env_v6(num_players=2, seed=None):
    config = {
        "allow_step_back": False,
        "seed": seed,
        "game_num_players": num_players,
    }
    return ExtendedNLHEnvV6(config)


class ExtendedNLHEnvV6(ExtendedNLHEnv):
    def __init__(self, config):
        super().__init__(config)
        self.state_shape = [[54 + NUM_EXTRA_V6] for _ in range(self.num_players)]

    def _extract_state(self, state):
        extracted = super()._extract_state(state)  # 68次元まで作られる

        eq = cached_equity(state["hand"], state["public_cards"],
                           self.num_players - 1)
        pot = float(state["pot"])
        my_stack = float(state["stakes"][state["current_player"]])
        spr = min(my_stack / pot, 20.0) / 20.0 if pot > 0 else 1.0

        extracted["obs"] = np.concatenate(
            [extracted["obs"], np.array([eq, spr], dtype=np.float64)])
        return extracted
