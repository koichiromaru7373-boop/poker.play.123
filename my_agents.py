"""my_agents.py - 自作エージェント置き場(importして使う部品箱)"""
import random

from rlcard.games.nolimitholdem.round import Action

RANKS = '23456789TJQKA'

def preflop_score(hand):
    r1, r2 = RANKS.index(hand[0][1]), RANKS.index(hand[1][1])
    hi, lo = max(r1, r2), min(r1, r2)
    suited = hand[0][0] == hand[1][0]
    score = hi * 0.5
    if r1 == r2:
        score = max(score * 2, 5)
    if suited:
        score += 1
    gap = hi - lo
    if gap == 1: score += 1
    elif gap >= 4: score -= 2
    return score

class RuleBasedAgent:
    use_raw = True

    def __init__(self, num_actions=None):
        self.num_actions = num_actions

    def step(self, state):
        raw = state['raw_obs']
        legal = state['raw_legal_actions']
        hand = raw['hand']
        board = raw['public_cards']

        if len(board) == 0:
            score = preflop_score(hand)
            if score >= 8 and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if score >= 4 and Action.CHECK_CALL in legal:
                return Action.CHECK_CALL
            if Action.CHECK_CALL in legal and raw['all_chips'][0] == raw['all_chips'][1]:
                return Action.CHECK_CALL
            return Action.FOLD
        else:
            my_ranks = [c[1] for c in hand]
            board_ranks = [c[1] for c in board]
            has_pair = (my_ranks[0] == my_ranks[1]
                        or my_ranks[0] in board_ranks
                        or my_ranks[1] in board_ranks)
            if has_pair and Action.RAISE_HALF_POT in legal:
                return Action.RAISE_HALF_POT
            if Action.CHECK_CALL in legal:
                return Action.CHECK_CALL
            return Action.FOLD

    def eval_step(self, state):
        return self.step(state), {}


class StyleAgent:
    """ルース度と攻撃性で性格を変えられるボット(ペルソナ用)

    looseness: 0-1。高いほど広いハンドで参加する(ルース)
    aggression: 0-1。高いほどレイズが増える(アグレッシブ)
    """
    use_raw = True

    def __init__(self, num_actions=None, looseness=0.5, aggression=0.5, seed=None):
        self.num_actions = num_actions
        self.looseness = looseness
        self.aggression = aggression
        self.rng = random.Random(seed)

    def _postflop_strength(self, hand, board):
        my_ranks = [c[1] for c in hand]
        board_ranks = [c[1] for c in board]
        pocket = my_ranks[0] == my_ranks[1]
        hits = sum(r in board_ranks for r in my_ranks)
        if hits >= 2 or (pocket and RANKS.index(my_ranks[0])
                         >= max(RANKS.index(b) for b in board_ranks)):
            return 2
        if hits == 1 or pocket:
            return 1
        return 0

    def step(self, state):
        raw = state['raw_obs']
        legal = state['raw_legal_actions']
        hand, board = raw['hand'], raw['public_cards']
        r = self.rng.random()

        def first(*acts):
            for a in acts:
                if a in legal:
                    return a
            return legal[0]

        can_check = raw['my_chips'] == max(raw['all_chips'])

        if len(board) == 0:
            score = preflop_score(hand)
            strong = score >= 8 - 3 * self.looseness    # ルースなほど基準が緩い
            playable = score >= 5 - 3 * self.looseness
        else:
            s = self._postflop_strength(hand, board)
            strong = s == 2
            playable = s >= 1

        if strong:
            if r < self.aggression:
                return first(Action.RAISE_POT, Action.RAISE_HALF_POT, Action.CHECK_CALL)
            return first(Action.CHECK_CALL)
        if playable:
            if r < self.aggression * 0.6:
                return first(Action.RAISE_HALF_POT, Action.CHECK_CALL)
            if can_check or r < 0.3 + 0.6 * self.looseness:
                return first(Action.CHECK_CALL)
            return first(Action.FOLD, Action.CHECK_CALL)
        # 弱いハンド
        if r < self.looseness * self.aggression * 0.35:  # たまにブラフ
            return first(Action.RAISE_HALF_POT, Action.CHECK_CALL)
        if can_check:
            return first(Action.CHECK_CALL)
        if r < self.looseness * 0.5:                     # ルースなコール
            return first(Action.CHECK_CALL)
        return first(Action.FOLD, Action.CHECK_CALL)

    def eval_step(self, state):
        return self.step(state), {}

