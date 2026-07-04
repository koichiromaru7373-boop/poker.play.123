"""GTOベースのヒント生成

プリフロップ: 公知のGTO定石に基づくハンドティア表で判断(説明文付き)
ポストフロップ: 役の強さを分類して説明文を作り、アクションは学習モデルの
               推奨を併記する(モデルはapi側から渡す)
"""

RANKS = '23456789TJQKA'

FOLD, CHECK_CALL, RAISE_HALF_POT, RAISE_POT, ALL_IN = 0, 1, 2, 3, 4

# プリフロップのハンドティア(1が最強)。表記: ペア"AA"、スーテッド"AKs"、オフ"AKo"
TIERS = {
    1: "AA KK QQ AKs",
    2: "JJ TT AKo AQs AJs KQs",
    3: "99 88 AQo ATs KJs QJs JTs",
    4: "77 66 55 AJo KQo A9s A8s A7s A6s A5s KTs QTs J9s T9s 98s",
    5: "44 33 22 ATo KJo QJo JTo A4s A3s A2s K9s Q9s T8s 87s 76s 65s",
    6: "A9o KTo QTo J9o T9o K8s Q8s J8s 97s 86s 75s 54s",
}
_TIER_OF = {}
for _t, _hands in TIERS.items():
    for _h in _hands.split():
        _TIER_OF[_h] = _t


def hand_code(hand):
    """["SA","HK"] -> "AKo" / ["SA","SK"] -> "AKs" / ["SA","HA"] -> "AA" """
    r1, r2 = hand[0][1], hand[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if hand[0][0] == hand[1][0] else "o")


def _first(legal, *acts):
    for a in acts:
        if a in legal:
            return a
    return legal[0]


def preflop_hint(raw, legal):
    """(action_id, 説明文) を返す"""
    code = hand_code(raw["hand"])
    tier = _TIER_OF.get(code, 7)
    pot = float(raw["pot"])
    facing_raise = pot > 3  # ブラインド(1+2)より大きい=誰かがレイズ済み
    can_check = raw["my_chips"] == max(raw["all_chips"])

    if tier <= 2:
        act = _first(legal, RAISE_POT, RAISE_HALF_POT, CHECK_CALL)
        why = "プレミアムハンドです。強く打って育てるのがGTOの定石(レイズ/3ベット推奨)"
    elif tier == 3:
        if facing_raise:
            act = _first(legal, CHECK_CALL)
            why = "強いハンドですが、レイズを受けた場合はコールで様子見が標準です"
        else:
            act = _first(legal, RAISE_HALF_POT, RAISE_POT, CHECK_CALL)
            why = "オープンレイズに十分な強さです。先制して主導権を取りましょう"
    elif tier <= 5:
        if facing_raise:
            act = _first(legal, FOLD, CHECK_CALL) if not can_check else CHECK_CALL
            why = "中程度のハンドはレイズに付き合うと損しやすい形です(フォールド寄り)"
        else:
            act = _first(legal, CHECK_CALL)
            why = "参加してもよい強さですが、大きなポットにする手ではありません"
    elif tier == 6:
        if can_check:
            act = CHECK_CALL
            why = "無料で見られるならフロップを見る価値はあります"
        else:
            act = _first(legal, FOLD, CHECK_CALL)
            why = "投機的なハンドです。チップを払ってまで参加する手ではありません"
    else:
        if can_check:
            act = CHECK_CALL
            why = "弱いハンドですが、チェックできるなら無料で様子を見ましょう"
        else:
            act = _first(legal, FOLD, CHECK_CALL)
            why = "GTOの参加レンジ外です。ここは降りるのが長期的に得です"

    return act, f"あなたの {code} はティア{min(tier, 7)}。{why}"


class GTOHybridAgent:
    """プリフロップ=GTO表、ポストフロップ=学習モデル のハイブリッド

    学習モデルをラップして使う。返すアクションは常にID(int)に統一。
    """
    use_raw = False

    def __init__(self, model_agent):
        self.model = model_agent

    @staticmethod
    def _to_id(action):
        return int(getattr(action, "value", action))

    def eval_step(self, state):
        raw = state["raw_obs"]
        if not raw["public_cards"]:  # プリフロップはGTO表に従う
            legal = [int(a) for a in state["legal_actions"].keys()]
            act, _ = preflop_hint(raw, legal)
            return act, {}
        action, info = self.model.eval_step(state)
        return self._to_id(action), info

    def step(self, state):
        return self.eval_step(state)[0]


def postflop_note(raw):
    """役の説明文(アクションはモデル側に任せる)"""
    hand, board = raw["hand"], raw["public_cards"]
    my_ranks = [c[1] for c in hand]
    board_ranks = [c[1] for c in board]
    pocket = my_ranks[0] == my_ranks[1]
    hits = sum(r in board_ranks for r in my_ranks)

    suits = [c[0] for c in hand] + [c[0] for c in board]
    flush_draw = any(suits.count(s) == 4 for s in set(suits))

    if hits >= 2:
        note = "ツーペア以上の強い役です。バリューを取りにいく場面"
    elif pocket and board_ranks and RANKS.index(my_ranks[0]) >= max(RANKS.index(b) for b in board_ranks):
        note = "オーバーペアです。強気に打てる場面"
    elif hits == 1:
        top = max(board_ranks, key=RANKS.index) in my_ranks
        note = "トップペアです。バリューベット対象" if top else "ワンペアですが、上のカードに注意"
    elif pocket:
        note = "ポケットペアですが、ボードに上のカードがあれば慎重に"
    elif flush_draw:
        note = "フラッシュドローがあります。安く見られるなら続行の価値あり"
    else:
        note = "現状ノーヒットです。無理せず降りることも検討を"
    return note
