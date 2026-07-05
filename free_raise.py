"""任意額レイズを rlcard(no-limit-holdem)に差し込むための薄いパッチ。

方針:
- `NolimitholdemRound.proceed_round` をパッチし、通常の Action(離散5種)に
  加えて `RaiseTo(added)` という「今から added チップ上乗せする」アクションを
  受け付けるようにする。
- ポインタ送り・降り/オールインのスキップ処理は **元の実装に丸投げ**する
  (レイズ直後の自分に対して no-op の CHECK_CALL を元関数で回し、その副産物の
   ポインタ送りだけを流用)。これにより rlcard のバージョン差(内部の細かい
   属性名やストリート遷移・オールインのランナウト処理)に影響されにくい。
- AI・ヒント・勝率は一切触らない(離散のまま)。人間だけが任意額を打てる。

使い方(api_server 側):
    import free_raise
    free_raise.enable()                       # 起動時に1回
    ...
    free_raise.apply_raise_to(env, target)    # target = レイズ後の自分の総 in_chips(ゲーム単位)
    b = free_raise.raise_bounds(env)          # {"to_call","min_to","max_to","my_in","pot"}

注意: rlcard 1.2.0 での動作確認は test_free_raise.py で行うこと。
"""

from rlcard.games.nolimitholdem.round import NolimitholdemRound

try:
    from rlcard.games.nolimitholdem.round import Action
except ImportError:  # 念のためのフォールバック
    from rlcard.games.nolimitholdem.game import Action


class RaiseTo:
    """任意額レイズを表すアクション擬態オブジェクト。

    added: 今から上乗せするチップ(ゲーム内単位)。
    合法性チェック(game.step 内の `action not in legal_actions`)を
    通すため、レイズ系 Action と == で一致するようにしてある。
    """

    __slots__ = ("added",)

    def __init__(self, added):
        self.added = int(added)

    def __eq__(self, other):
        return other in (Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN)

    def __hash__(self):
        return hash(Action.RAISE_POT)

    def __repr__(self):
        return f"RaiseTo(added={self.added})"


# パッチ前の元メソッドを保持(再入・二重パッチ対策で import 時に確保)
_ORIG_PROCEED = NolimitholdemRound.proceed_round
_ENABLED = False


def _patched_proceed(self, players, action):
    if isinstance(action, RaiseTo):
        gp = self.game_pointer
        player = players[gp]
        added = action.added

        # 1) 実際に上乗せ(元コードと同じプリミティブ: raised[] と player.bet)
        self.raised[gp] += added
        player.bet(added)

        # 2) ポインタ送り/スキップは元実装に委譲する。
        #    いま自分は場の最大額なので、元の CHECK_CALL は差分0の no-op になり、
        #    「次に打つ人へのポインタ送り」だけが得られる。
        new_gp = _ORIG_PROCEED(self, players, Action.CHECK_CALL)

        # 3) レイズなのでアクションを再オープン(元の raise 分岐と同じ値に上書き)
        self.not_raise_num = 1

        # 4) 全ツッパならオールイン状態に(元実装が付けていなければ補完)
        if player.remained_chips == 0 and getattr(player.status, "name", "") not in ("FOLDED", "ALLIN"):
            try:
                player.status = type(player.status).ALLIN
            except Exception:
                pass
        return new_gp

    return _ORIG_PROCEED(self, players, action)


def enable():
    """パッチを適用する(冪等)。api_server 起動時に1回呼ぶ。"""
    global _ENABLED
    if not _ENABLED:
        NolimitholdemRound.proceed_round = _patched_proceed
        _ENABLED = True


# ---- API から使う補助関数 ----

def raise_bounds(env):
    """現在の手番プレイヤーのレイズ可能レンジ(ゲーム内チップ単位)を返す。

    返り値(すべて int, ゲーム内単位):
      to_call : コールに必要な額
      min_to  : ミニマムレイズ後の総 in_chips(目安)
      max_to  : オールイン時の総 in_chips
      my_in   : 現在の自分の総 in_chips
      pot     : 現在のポット(全員の in_chips 合計)
    """
    g = env.game
    pid = g.get_player_id()
    players = g.players
    me = players[pid]
    in_chips = [int(p.in_chips) for p in players]
    my_in = int(me.in_chips)
    my_remained = int(me.remained_chips)
    highest = max(in_chips)

    to_call = highest - my_in
    max_to = my_in + my_remained  # オールイン
    bb = int(getattr(g, "big_blind", 2) or 2)
    # ミニマムレイズ: 現在の最大額 + 1BB(近似)。オールインで上限。
    min_to = min(highest + bb, max_to)
    return {
        "to_call": to_call,
        "min_to": min_to,
        "max_to": max_to,
        "my_in": my_in,
        "pot": sum(in_chips),
    }


def apply_raise_to(env, target):
    """手番プレイヤーを「総 in_chips = target(ゲーム内単位)」までレイズさせる。

    target がレンジ外・レイズ未満なら ValueError。
    """
    b = raise_bounds(env)
    target = int(target)

    highest = b["my_in"] + b["to_call"]  # = 現在の場の最大額
    if target > b["max_to"]:
        target = b["max_to"]  # オールインに丸め
    if target <= highest:
        raise ValueError("レイズにはコール額を上回る必要があります(コール/オールインを使ってください)")
    if target < b["min_to"]:
        raise ValueError(f"レイズ額が小さすぎます(最小 {b['min_to']} まで上げてください)")

    added = target - b["my_in"]
    if added <= 0:
        raise ValueError("レイズになっていません")

    env.game.step(RaiseTo(added))
    return target
