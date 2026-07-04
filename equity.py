"""勝率(Equity)推定と役名判定

rlcardの7枚役評価器(limitholdem.utils)を流用したモンテカルロ推定。
API応答用(数百試行で約50〜150ms)。学習組み込み(v6)でも使う予定。
"""

import random

from rlcard.games.limitholdem.utils import Hand, compare_hands

FULL_DECK = [s + r for s in "SHDC" for r in "23456789TJQKA"]

RANK_NAMES_JA = {
    1: "ハイカード",
    2: "ワンペア",
    3: "ツーペア",
    4: "スリーカード",
    5: "ストレート",
    6: "フラッシュ",
    7: "フルハウス",
    8: "フォーカード",
    9: "ストレートフラッシュ",
}


def hand_rank_name(hole_cards, public_cards):
    """7枚(2+5)の役名を返す。5枚未満のボードではNone"""
    if len(public_cards) < 5:
        return None
    try:
        h = Hand(list(hole_cards) + list(public_cards))
        h.evaluateHand()
        return RANK_NAMES_JA.get(h.category, None)
    except Exception:
        return None


def estimate_win_prob(hole_cards, public_cards, num_opponents=1,
                      num_samples=300, rng=None):
    """モンテカルロで勝率を推定(スプリットは按分)"""
    rng = rng or random
    known = set(hole_cards) | set(public_cards)
    deck = [c for c in FULL_DECK if c not in known]
    need_board = 5 - len(public_cards)
    wins = 0.0
    for _ in range(num_samples):
        drawn = rng.sample(deck, num_opponents * 2 + need_board)
        board = list(public_cards) + drawn[num_opponents * 2:]
        hands = [list(hole_cards) + board]
        for i in range(num_opponents):
            hands.append(drawn[i * 2:i * 2 + 2] + board)
        try:
            result = compare_hands(hands)
        except Exception:
            continue
        if result[0] == 1:
            wins += 1.0 / sum(result)
    return round(wins / num_samples, 3)
