"""拡張観測付き no-limit-holdem 環境 (v5系)

標準の54次元(カード52+チップ2)に、人間なら必ず見る情報14次元を追加する:
- ポットサイズ(正規化)
- コールに必要な額(正規化)
- ポットオッズ
- 自分の残りスタック(正規化)
- ストリート one-hot 4 (プリフロップ/フロップ/ターン/リバー)
- 自分の席番号 one-hot 6

※この環境で学習したモデルは観測次元が違うため、旧モデル(v2〜v4, NFSP)とは
  互換性がない。デプロイ時もこの環境を使うこと。

あわせて、プリフロップをGTO表で固定するエージェントラッパー
`PreflopGTO` もここに置く(学習をポストフロップに集中させるため)。
"""

import numpy as np

from rlcard.envs.nolimitholdem import NolimitholdemEnv

from gto_hints import preflop_hint

NUM_EXTRA = 14
STACK_NORM = 100.0  # rlcard初期スタック


def make_extended_env(num_players=2, seed=None):
    config = {
        "allow_step_back": False,
        "seed": seed,
        "game_num_players": num_players,
    }
    return ExtendedNLHEnv(config)


class ExtendedNLHEnv(NolimitholdemEnv):
    def __init__(self, config):
        super().__init__(config)
        self.state_shape = [[54 + NUM_EXTRA] for _ in range(self.num_players)]

    def _extract_state(self, state):
        extracted = super()._extract_state(state)
        extra = np.zeros(NUM_EXTRA, dtype=np.float64)

        pot = float(state["pot"])
        my_bet = float(state["my_chips"])
        to_call = float(max(state["all_chips"])) - my_bet
        my_stack = float(state["stakes"][state["current_player"]])

        extra[0] = pot / (STACK_NORM * self.num_players)
        extra[1] = to_call / STACK_NORM
        extra[2] = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        extra[3] = my_stack / STACK_NORM

        stage_i = min(int(getattr(state["stage"], "value", 0)), 3)
        extra[4 + stage_i] = 1.0

        seat = int(state["current_player"]) % 6
        extra[8 + seat] = 1.0

        extracted["obs"] = np.concatenate([extracted["obs"], extra])
        return extracted


class PreflopGTO:
    """プリフロップ=GTO表、ポストフロップ=中身のエージェント

    学習用ラッパー。返すアクションは常にID(int)。
    """
    use_raw = False

    def __init__(self, inner):
        self.inner = inner

    @staticmethod
    def _to_id(action):
        return int(getattr(action, "value", action))

    def step(self, state):
        raw = state["raw_obs"]
        if not raw["public_cards"]:
            legal = [int(a) for a in state["legal_actions"].keys()]
            act, _ = preflop_hint(raw, legal)
            return act
        return self._to_id(self.inner.step(state))

    def eval_step(self, state):
        raw = state["raw_obs"]
        if not raw["public_cards"]:
            legal = [int(a) for a in state["legal_actions"].keys()]
            act, _ = preflop_hint(raw, legal)
            return act, {}
        action, info = self.inner.eval_step(state)
        return self._to_id(action), info


def is_postflop_transition(ts):
    """trajectory の遷移がポストフロップの判断かどうか"""
    return bool(ts[0]["raw_obs"]["public_cards"])
