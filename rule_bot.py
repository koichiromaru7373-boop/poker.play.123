"""
rule_bot.py - Phase 2: ルールベースボット vs ランダムボット
プリフロップはハンドレンジ表、ポストフロップは簡易ロジックで判断する
"""
import rlcard
from rlcard.agents import RandomAgent
from rlcard.games.nolimitholdem.round import Action

RANKS = '23456789TJQKA'

def preflop_score(hand):
    """2枚のハンドを 0-10 点で評価する簡易スコア(Chen formula風)"""
    r1, r2 = RANKS.index(hand[0][1]), RANKS.index(hand[1][1])
    hi, lo = max(r1, r2), min(r1, r2)
    suited = hand[0][0] == hand[1][0]
    score = hi * 0.5                      # 高いカードほど加点
    if r1 == r2:
        score = max(score * 2, 5)         # ポケットペアは強い
    if suited:
        score += 1                        # スーテッド加点
    gap = hi - lo
    if gap == 1: score += 1               # コネクター加点
    elif gap >= 4: score -= 2             # 離れすぎは減点
    return score

class RuleBasedAgent:
    """RLCardのエージェント規約: step() と eval_step() を持つクラス"""
    use_raw = True  # raw_obs(人間語の情報)を受け取るモード

    def step(self, state):
        raw = state['raw_obs']
        legal = state['raw_legal_actions']
        hand = raw['hand']
        board = raw['public_cards']

        if len(board) == 0:
            # --- プリフロップ: ハンドスコアで3段階判断 ---
            score = preflop_score(hand)
            if score >= 8 and Action.RAISE_POT in legal:
                return Action.RAISE_POT          # プレミアム: 攻める
            if score >= 4 and Action.CHECK_CALL in legal:
                return Action.CHECK_CALL         # 並: 見る
            if Action.CHECK_CALL in legal and raw['all_chips'][0] == raw['all_chips'][1]:
                return Action.CHECK_CALL         # タダで見られるならチェック
            return Action.FOLD                   # ゴミ: 投げる
        else:
            # --- ポストフロップ: ペア以上ができたかだけ見る簡易版 ---
            my_ranks = [c[1] for c in hand]
            board_ranks = [c[1] for c in board]
            has_pair = (my_ranks[0] == my_ranks[1]           # ポケットペア
                        or my_ranks[0] in board_ranks         # ボードとヒット
                        or my_ranks[1] in board_ranks)
            if has_pair and Action.RAISE_HALF_POT in legal:
                return Action.RAISE_HALF_POT
            if Action.CHECK_CALL in legal:
                return Action.CHECK_CALL
            return Action.FOLD

    def eval_step(self, state):
        return self.step(state), {}

# --- 対戦実験: ルールボット vs ランダムボット ---
env = rlcard.make('no-limit-holdem', config={'seed': 42})
env.set_agents([RuleBasedAgent(), RandomAgent(num_actions=env.num_actions)])

NUM_HANDS = 1000
total = 0
for _ in range(NUM_HANDS):
    _, payoffs = env.run(is_training=False)
    total += payoffs[0]

print("=" * 40)
print(f"対戦結果 ({NUM_HANDS}ハンド)")
print(f"ルールボットの収支: {total:+.1f} BB")
print(f"平均: {total/NUM_HANDS*100:+.1f} BB/100ハンド")
print("=" * 40)
