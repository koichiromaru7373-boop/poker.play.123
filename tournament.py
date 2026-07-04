"""トーナメント進行ラッパー (Phase 5)

RLCardの1ハンドエンジンの上に、以下を実装する:
- チップのハンド間持ち越し
- ブラインド上昇(--blind_up_every ハンドごとに倍)
  ※RLCard内部のブラインドはSB=1/BB=2で固定のため、
    「実チップをブラインド単位に換算してから渡す」方式で等価に実現する
- 0チップになった席は脱落。最後の1人が優勝

使い方(AI同士のシミュレーションは test_tournament.py を参照):
    t = Tournament(agents={0: user_placeholder, 1: ai, ...}, starting_stack=200)
    while not t.finished:
        t.play_hand(decide_fn=...)  # decide_fnで人間の手番を差し込める
"""

import random

import rlcard

BASE_BB = 2  # rlcard内部の固定ビッグブラインド


class Tournament:
    def __init__(self, agents, starting_stack=200, blind_up_every=10, seed=None):
        """agents: {席番号: エージェント}。席0をユーザーにする想定(AIのみでも可)"""
        if seed is not None:
            random.seed(seed)
        self.agents = dict(agents)
        self.seats = sorted(self.agents)
        self.stacks = {s: int(starting_stack) for s in self.seats}
        self._total_chips = int(starting_stack) * len(self.seats)
        self.blind_up_every = blind_up_every
        self.hand_no = 0
        self.finished = False
        self.winner = None
        self.button_pos = random.randrange(len(self.seats))
        self.last_hand = None  # 直近ハンドの記録(表示用)
        self.current = None    # 進行中ハンド {"env", "alive", "unit"}

    # ---- 状態 ----
    @property
    def level(self):
        return self.hand_no // self.blind_up_every

    @property
    def big_blind(self):
        """現在の実ビッグブラインド額"""
        return BASE_BB * (2 ** self.level)

    def alive(self):
        return [s for s in self.seats if self.stacks[s] > 0]

    def standings(self):
        return {
            "hand_no": self.hand_no,
            "big_blind": self.big_blind,
            "stacks": dict(self.stacks),
            "alive": self.alive(),
            "finished": self.finished,
            "winner": self.winner,
        }

    # ---- 進行 ----
    def start_hand(self):
        """次のハンドをセットアップして環境を返す(API用に途中停止可能な形)"""
        alive = self.alive()
        if len(alive) <= 1:
            self.finished = True
            self.winner = alive[0] if alive else None
            return None

        self.hand_no += 1
        unit = self.big_blind / BASE_BB  # 実チップ1単位 = ゲーム内 1/unit チップ

        env = rlcard.make(
            "no-limit-holdem",
            config={"game_num_players": len(alive),
                    "seed": random.randrange(1 << 30)},
        )
        # 実チップをブラインド換算してRLCard内部に直接セットする
        game_chips = [max(1, int(self.stacks[s] / unit)) for s in alive]
        env.game.init_chips = game_chips
        # ボタン(ディーラー)を回す。rlcard側に属性があれば固定、なければランダムのまま
        self.button_pos = (self.button_pos + 1) % len(alive)
        if hasattr(env.game, "dealer_id"):
            env.game.dealer_id = self.button_pos

        env.reset()
        self.current = {"env": env, "alive": alive, "unit": unit}
        return self.current

    def settle(self):
        """終了したハンドの精算(チップ反映・脱落・優勝判定)"""
        cur = self.current
        env, alive, unit = cur["env"], cur["alive"], cur["unit"]
        payoffs = env.get_payoffs()  # ゲーム内チップ単位の増減
        for i, s in enumerate(alive):
            delta = int(round(float(payoffs[i]) * unit))
            self.stacks[s] = max(0, self.stacks[s] + delta)

        # 端数でチップ総量がずれた場合はチップリーダーに寄せて保存則を守る
        total = sum(self.stacks.values())
        drift = self._total_chips - total
        if drift != 0:
            top = max(self.alive(), key=lambda s: self.stacks[s])
            self.stacks[top] += drift

        self.last_hand = {
            "hand_no": self.hand_no,
            "big_blind": self.big_blind,
            "payoffs_game_units": [float(p) for p in payoffs],
            "seats_in_hand": alive,
        }
        self.current = None

        alive_now = self.alive()
        if len(alive_now) <= 1:
            self.finished = True
            self.winner = alive_now[0] if alive_now else None
        return self.last_hand

    def play_hand(self, decide_fn=None, verbose=False):
        """1ハンドを最後まで進める(シミュレーション用)。

        decide_fn(seat, state) -> action を渡すと、その席はAIではなく
        decide_fnの返すアクションで打つ。Noneなら全席AI。
        """
        cur = self.start_hand()
        if cur is None:
            return None
        env, alive = cur["env"], cur["alive"]
        while not env.is_over():
            pid = env.get_player_id()
            seat = alive[pid]
            state = env.get_state(pid)
            action = decide_fn(seat, state) if decide_fn is not None else None
            if action is None:
                action, _ = self.agents[seat].eval_step(state)
            env.step(action)
        result = self.settle()

        if verbose:
            print(f"hand {self.hand_no:>3} BB={self.big_blind:>4} "
                  f"stacks={ {s: self.stacks[s] for s in self.seats} }"
                  f"{'  WINNER: seat ' + str(self.winner) if self.finished else ''}")
        return result
