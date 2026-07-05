"""Phase 5: ポーカー対戦API (FastAPI) — 雛形

ゲーム進行(ディーラー・ポット・役判定)はすべてAPI側が持つ。
サイト側は「状態を表示し、ユーザーのアクションを送る」だけ。

現状は1ハンド完結(RLCardのまま)。チップ持ち越し・ブラインド上昇・
脱落(トーナメント)は次段階でラッパーを実装する。

起動:
    pip install fastapi uvicorn
    uvicorn api_server:app --reload

エンドポイント:
    POST /tables                     テーブル作成 {"mode": "heads_up"|"six_max", "level": "beginner"|"intermediate"|"advanced"}
    GET  /tables/{table_id}          現在の状態(ユーザー視点)
    POST /tables/{table_id}/action   ユーザーのアクション {"action": 0-4}
    GET  /tables/{table_id}/hint     AIの推奨アクション(打ち方を学べる機能)
    POST /tables/{table_id}/next     ハンド終了後、次のハンドを開始
"""

import os
import uuid
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import rlcard

import free_raise
from equity import estimate_win_prob, hand_rank_name
from extended_env import make_extended_env
from gto_hints import GTOHybridAgent, postflop_note, preflop_hint
from my_agents import RuleBasedAgent
from personas import build_tournament_agents
from tournament import Tournament

# 任意額レイズ(人間のみ)を有効化。AI・ヒント・勝率は離散のまま。
free_raise.enable()

NFSP_PATH = os.path.join("experiments", "nfsp", "nfsp_model_best.pth")


def _hint_agent(mode):
    """ヒント役: ヘッズアップはNFSPがあれば優先(均衡戦略に近いため)"""
    if mode == "heads_up" and os.path.exists(NFSP_PATH):
        try:
            return torch.load(NFSP_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(NFSP_PATH, map_location="cpu")
    return load_ai(mode, "advanced", 5)


def make_hint(mode, state, agent=None):
    """プリフロップ=GTO表、ポストフロップ=学習モデル+役分析

    agent指定時はそれを使う(拡張観測テーブルでは観測次元が合う
    テーブル自身のAIを渡すこと)
    """
    raw = state["raw_obs"]
    legal = [int(a) for a in state["legal_actions"].keys()]
    if not raw["public_cards"]:
        act, reason = preflop_hint(raw, legal)
        source = "GTOプリフロップ表"
    else:
        if agent is None:
            agent = _hint_agent(mode)
        action, _ = agent.eval_step(state)
        act = action_id(action)
        reason = postflop_note(raw)
        source = "学習モデル+役分析"
    return {
        "recommended_action": {"id": act, "name": ACTION_NAMES[act]},
        "reason": reason,
        "source": source,
    }


def step_env(env, agent_action, agent=None):
    """use_raw なエージェント(Action列挙を返す)にも対応して環境を進める"""
    raw = bool(getattr(agent, "use_raw", False)) if agent is not None else False
    env.step(agent_action, raw_action=raw)


def action_id(action):
    """Action列挙でもintでもアクションIDに揃える"""
    return int(getattr(action, "value", action))


def capture_reveal(env, seat_of):
    """ハンド終了時の開示情報(全ハンド・役名・勝者)を取得する

    seat_of: プレイヤーID(env内の席順) -> 表示上の席番号
    """
    try:
        payoffs = env.get_payoffs()
        public = [c.get_index() for c in env.game.public_cards]
        hands, ranks = {}, {}
        for pid, p in enumerate(env.game.players):
            status = getattr(p.status, "name", str(p.status))
            if "FOLDED" in status:
                continue
            hole = [c.get_index() for c in p.hand]
            hands[str(seat_of[pid])] = hole
            rank = hand_rank_name(hole, public)
            if rank:
                ranks[str(seat_of[pid])] = rank
        winners = [seat_of[pid] for pid in range(len(env.game.players))
                   if float(payoffs[pid]) > 0]
        # ボードは手札と同じ get_index() 形式(suit+rank)で返す。
        # ※ str(card) は rank+suit を返すためフロントで表示が壊れる
        if len(hands) < 2:
            hands, ranks = {}, {}  # 全員降りで決着: ハンド開示なし
        return {"hands": hands, "rank_names": ranks, "winners": winners,
                "public_cards": public}
    except Exception:
        return None


def count_active_opponents(env, my_pid):
    n = 0
    for pid, p in enumerate(env.game.players):
        status = getattr(p.status, "name", str(p.status))
        if pid != my_pid and "FOLDED" not in status:
            n += 1
    return max(1, n)

# ---- 難易度 → モデルの対応 ----
# models/ の軽量版(デプロイ用)を最優先、無ければ experiments/ の学習出力を使う
def _pick(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None  # None = RuleBasedAgent


SLIM_ADV = os.path.join("models", "heads_up_advanced.pth")
V5_BEST = os.path.join("experiments", "dqn_v5", "dqn_model_best.pth")
MODEL_PATHS = {
    "heads_up": {
        "beginner": None,  # None = RuleBasedAgent
        "intermediate": _pick(os.path.join("models", "heads_up_intermediate.pth"),
                              os.path.join("experiments", "dqn_v2", "dqn_model_best.pth")),
        "advanced": _pick(SLIM_ADV, V5_BEST,
                          os.path.join("experiments", "dqn_v4", "dqn_model_best.pth")),
    },
    "six_max": {
        "beginner": None,
        "intermediate": _pick(os.path.join("models", "six_max.pth"),
                              os.path.join("experiments", "dqn_6max", "dqn6_model_best.pth")),
        "advanced": _pick(os.path.join("models", "six_max.pth"),
                          os.path.join("experiments", "dqn_6max", "dqn6_model_best.pth")),
    },
}
# 拡張観測(68次元)が必要なモデル
EXTENDED_MODELS = {SLIM_ADV, V5_BEST}

ACTION_NAMES = {0: "fold", 1: "check_call", 2: "raise_half_pot", 3: "raise_pot", 4: "all_in"}
USER_SEAT = 0

app = FastAPI(title="Poker AI API")
tables = {}  # table_id -> dict(env, ai_agents, mode, level, finished, payoffs)

# 環境変数 ACCESS_TOKEN を設定すると、?token=<値> が無いアクセスを拒否する
# (クラウド公開時の簡易ガード。未設定ならチェックなし=ローカル用)
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")


@app.middleware("http")
async def token_guard(request, call_next):
    if ACCESS_TOKEN:
        token = (request.query_params.get("token")
                 or request.headers.get("x-access-token"))
        if token != ACCESS_TOKEN:
            return JSONResponse({"detail": "アクセストークンが必要です"}, status_code=401)
    return await call_next(request)


@app.get("/play")
def play_page():
    """ブラウザで遊べるテスト用画面"""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "play.html"))


def load_ai(mode, level, num_actions):
    path = MODEL_PATHS[mode][level]
    if path is None or not os.path.exists(path):
        return RuleBasedAgent(num_actions=num_actions)  # 初級はあえて弱いまま
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    # 学習モデルはGTOハイブリッド化(プリフロップの穴を塞ぐ)
    return GTOHybridAgent(model)


class CreateTable(BaseModel):
    mode: str = "heads_up"          # heads_up | six_max
    level: str = "intermediate"     # beginner | intermediate | advanced


class UserAction(BaseModel):
    action: int  # 0=fold 1=check/call 2=raise half pot 3=raise pot 4=all-in
    # amount 指定時は任意額レイズ(= レイズ後の自分の総 in_chips, ゲーム内単位)。
    # 未指定なら従来通り action の離散アクションで打つ。
    amount: Optional[int] = None


def advance_ai(table):
    """ユーザーの手番になるかハンドが終わるまでAIを打たせる"""
    env = table["env"]
    while not env.is_over():
        pid = env.get_player_id()
        if pid == USER_SEAT:
            return
        state = env.get_state(pid)
        agent = table["ai_agents"][pid]
        action, _ = agent.eval_step(state)
        step_env(env, action, agent)
    table["finished"] = True
    table["payoffs"] = [float(p) for p in env.get_payoffs()]


def user_view(table_id):
    table = tables[table_id]
    env = table["env"]
    state = env.get_state(USER_SEAT)
    raw = state["raw_obs"]
    view = {
        "table_id": table_id,
        "mode": table["mode"],
        "level": table["level"],
        "hand": [str(c) for c in raw["hand"]],
        "public_cards": [str(c) for c in raw["public_cards"]],
        "pot": float(raw["pot"]),
        "my_chips": int(raw["my_chips"]),
        "all_chips": [int(c) for c in raw["all_chips"]],
        "stage": str(raw["stage"]),
        "finished": table["finished"],
    }
    if table["finished"]:
        view["payoffs"] = table["payoffs"]
        view["legal_actions"] = []
    else:
        view["legal_actions"] = [
            {"id": int(a), "name": ACTION_NAMES[int(a)]}
            for a in state["legal_actions"].keys()
        ]
        # 任意額レイズ用のレンジ(自分の手番のときだけ)
        if env.get_player_id() == USER_SEAT:
            try:
                b = free_raise.raise_bounds(env)
                view["to_call"] = b["to_call"]
                view["raise_min"] = b["min_to"]
                view["raise_max"] = b["max_to"]
            except Exception:
                pass
    return view


def new_hand(table):
    env = table["env"]
    env.reset()
    table["finished"] = False
    table["payoffs"] = None
    advance_ai(table)


@app.post("/tables")
def create_table(req: CreateTable):
    if req.mode not in MODEL_PATHS:
        raise HTTPException(400, "mode は heads_up か six_max")
    if req.level not in MODEL_PATHS[req.mode]:
        raise HTTPException(400, "level は beginner/intermediate/advanced")

    num_players = 2 if req.mode == "heads_up" else 6
    # v5系モデル(68次元)を使うテーブルは拡張観測環境で作る
    path = MODEL_PATHS[req.mode][req.level]
    use_extended = path in EXTENDED_MODELS and path is not None and os.path.exists(path)
    if use_extended:
        env = make_extended_env(num_players=num_players)
    else:
        env = rlcard.make("no-limit-holdem", config={"game_num_players": num_players})

    ai = load_ai(req.mode, req.level, env.num_actions)
    ai_agents = {seat: ai for seat in range(1, num_players)}  # 席1〜がAI

    table_id = str(uuid.uuid4())[:8]
    tables[table_id] = {
        "env": env, "ai_agents": ai_agents,
        "mode": req.mode, "level": req.level,
        "extended": use_extended,
        "finished": False, "payoffs": None,
    }
    new_hand(tables[table_id])
    return user_view(table_id)


@app.get("/tables/{table_id}")
def get_state(table_id: str):
    if table_id not in tables:
        raise HTTPException(404, "table not found")
    return user_view(table_id)


@app.post("/tables/{table_id}/action")
def post_action(table_id: str, req: UserAction):
    if table_id not in tables:
        raise HTTPException(404, "table not found")
    table = tables[table_id]
    env = table["env"]
    if table["finished"]:
        raise HTTPException(400, "ハンド終了済み。/next で次のハンドを開始")
    if env.get_player_id() != USER_SEAT:
        raise HTTPException(400, "ユーザーの手番ではありません")

    state = env.get_state(USER_SEAT)
    if req.amount is not None:
        # 任意額レイズ
        try:
            free_raise.apply_raise_to(env, req.amount)
        except ValueError as e:
            raise HTTPException(400, str(e))
    else:
        if req.action not in state["legal_actions"]:
            raise HTTPException(400, f"不正なアクション。legal: {list(state['legal_actions'].keys())}")
        env.step(req.action)

    if env.is_over():
        table["finished"] = True
        table["payoffs"] = [float(p) for p in env.get_payoffs()]
    else:
        advance_ai(table)
    return user_view(table_id)


@app.get("/tables/{table_id}/hint")
def get_hint(table_id: str):
    """打ち方を学べる機能: 上級AIならどう打つかを返す"""
    if table_id not in tables:
        raise HTTPException(404, "table not found")
    table = tables[table_id]
    env = table["env"]
    if table["finished"] or env.get_player_id() != USER_SEAT:
        raise HTTPException(400, "ユーザーの手番ではありません")

    state = env.get_state(USER_SEAT)
    hint_agent = None
    if table.get("extended"):  # 拡張観測テーブルは自卓のAI(次元が合う)で助言
        hint_agent = next(iter(table["ai_agents"].values()))
    return make_hint(table["mode"], state, agent=hint_agent)


@app.post("/tables/{table_id}/next")
def next_hand(table_id: str):
    if table_id not in tables:
        raise HTTPException(404, "table not found")
    table = tables[table_id]
    if not table["finished"]:
        raise HTTPException(400, "ハンドがまだ終わっていません")
    new_hand(table)
    return user_view(table_id)


# ==================== 6人トーナメント ====================

tournaments = {}  # tournament_id -> dict(t, ai, level, hand_finished)


class CreateTournament(BaseModel):
    level: str = "intermediate"     # beginner | intermediate | advanced
    starting_stack: int = 200
    blind_up_every: int = 10


class TournamentAction(BaseModel):
    action: int
    amount: Optional[int] = None  # 任意額レイズ(レイズ後の総 in_chips, ゲーム内単位)


def t_advance(rec):
    """ユーザーの手番になるかハンドが終わるまでAIを打たせる"""
    t = rec["t"]
    env, alive = t.current["env"], t.current["alive"]
    while not env.is_over():
        pid = env.get_player_id()
        seat = alive[pid]
        if seat == USER_SEAT:
            return
        state = env.get_state(pid)
        agent = t.agents[seat]
        action, _ = agent.eval_step(state)
        rec.setdefault("hand_actions", []).append(
            {"seat": seat, "action_id": action_id(action)})
        step_env(env, action, agent)
    rec["reveal"] = capture_reveal(env, alive)
    t.settle()
    rec["hand_finished"] = True


def t_start_hand(rec):
    t = rec["t"]
    rec["hand_finished"] = False
    rec["hand_actions"] = []
    rec["reveal"] = None
    if t.start_hand() is None:
        return
    # ユーザーが脱落済みなら観戦モード: 決着までAIだけで進める
    if USER_SEAT not in t.current["alive"]:
        while not t.finished:
            t_advance(rec)
            if not t.finished and t.start_hand() is None:
                break
        return
    t_advance(rec)


def t_view(tournament_id):
    rec = tournaments[tournament_id]
    t = rec["t"]
    view = {
        "tournament_id": tournament_id,
        "level": rec["level"],
        "seats": rec.get("seats_info", {}),
        "standings": t.standings(),
        "hand_actions": rec.get("hand_actions", []),
        "hand_finished": rec["hand_finished"],
        "last_hand": t.last_hand,
        "my_turn": False,
    }
    reveal = rec.get("reveal")
    if rec["hand_finished"] and reveal:
        view["showdown"] = reveal
        # 決着時の最終ボードを送る(これが無いとフロントがランナウトできない)
        if reveal.get("public_cards"):
            view["public_cards"] = reveal["public_cards"]
        # 自分がショーダウンまで残っていたら手札も返す
        #  → フロントの「自分が残っている時だけリバーまで演出」を有効化
        uh = reveal["hands"].get(str(USER_SEAT))
        if uh:
            view["hand"] = uh
        if str(USER_SEAT) not in reveal["hands"]:
            view["ai_hands_revealed"] = reveal["hands"]  # 降りたユーザーへの学習用開示
    if t.finished:
        return view
    if t.current is not None:
        env, alive = t.current["env"], t.current["alive"]
        if not env.is_over():
            # 全員に見える情報(ボード・ポット・各席のベット額)を常に返す
            pid = env.get_player_id()
            raw = env.get_state(pid)["raw_obs"]
            view["public_cards"] = [str(c) for c in raw["public_cards"]]
            view["pot"] = float(raw["pot"])
            view["stage"] = str(raw["stage"]).split(".")[-1]
            view["bets"] = {alive[i]: int(c) for i, c in enumerate(raw["all_chips"])}
            # 自分の手札(生きていれば手番でなくても見せる)
            if USER_SEAT in alive:
                upid = alive.index(USER_SEAT)
                uraw = env.get_state(upid)["raw_obs"]
                view["hand"] = [str(c) for c in uraw["hand"]]
            if alive[pid] == USER_SEAT:
                state = env.get_state(pid)
                view["my_turn"] = True
                view["legal_actions"] = [
                    {"id": int(a), "name": ACTION_NAMES[int(a)]}
                    for a in state["legal_actions"].keys()
                ]
                try:
                    b = free_raise.raise_bounds(env)
                    view["to_call"] = b["to_call"]
                    view["raise_min"] = b["min_to"]
                    view["raise_max"] = b["max_to"]
                except Exception:
                    pass
                view["win_probability"] = estimate_win_prob(
                    view.get("hand", []), view.get("public_cards", []),
                    num_opponents=count_active_opponents(env, pid),
                    num_samples=300)
    return view


@app.post("/tournaments")
def create_tournament(req: CreateTournament):
    if req.level not in MODEL_PATHS["six_max"]:
        raise HTTPException(400, "level は beginner/intermediate/advanced")
    ai_agents, seats_info = build_tournament_agents(
        lambda: load_ai("six_max", req.level, 5), num_actions=5)
    agents = {0: RuleBasedAgent(num_actions=5)}  # 席0はユーザー(実際には使われない)
    agents.update(ai_agents)
    t = Tournament(agents, starting_stack=req.starting_stack,
                   blind_up_every=req.blind_up_every)
    tournament_id = str(uuid.uuid4())[:8]
    tournaments[tournament_id] = {"t": t, "level": req.level,
                                  "seats_info": seats_info,
                                  "hand_finished": False}
    t_start_hand(tournaments[tournament_id])
    return t_view(tournament_id)


@app.get("/tournaments/{tournament_id}")
def get_tournament(tournament_id: str):
    if tournament_id not in tournaments:
        raise HTTPException(404, "tournament not found")
    return t_view(tournament_id)


@app.post("/tournaments/{tournament_id}/action")
def tournament_action(tournament_id: str, req: TournamentAction):
    if tournament_id not in tournaments:
        raise HTTPException(404, "tournament not found")
    rec = tournaments[tournament_id]
    t = rec["t"]
    if t.finished or t.current is None:
        raise HTTPException(400, "進行中のハンドがありません。/next を使ってください")
    env, alive = t.current["env"], t.current["alive"]
    pid = env.get_player_id()
    if alive[pid] != USER_SEAT:
        raise HTTPException(400, "ユーザーの手番ではありません")
    state = env.get_state(pid)
    if req.amount is not None:
        # 任意額レイズ
        try:
            target = free_raise.apply_raise_to(env, req.amount)
        except ValueError as e:
            raise HTTPException(400, str(e))
        rec.setdefault("hand_actions", []).append(
            {"seat": USER_SEAT, "action_id": 3, "amount": int(target)})
    else:
        if req.action not in state["legal_actions"]:
            raise HTTPException(400, f"不正なアクション。legal: {list(state['legal_actions'].keys())}")
        rec.setdefault("hand_actions", []).append(
            {"seat": USER_SEAT, "action_id": int(req.action)})
        env.step(req.action)

    if env.is_over():
        rec["reveal"] = capture_reveal(env, alive)
        t.settle()
        rec["hand_finished"] = True
    else:
        t_advance(rec)
    return t_view(tournament_id)


@app.get("/tournaments/{tournament_id}/hint")
def tournament_hint(tournament_id: str):
    if tournament_id not in tournaments:
        raise HTTPException(404, "tournament not found")
    rec = tournaments[tournament_id]
    t = rec["t"]
    if t.finished or t.current is None:
        raise HTTPException(400, "ユーザーの手番ではありません")
    env, alive = t.current["env"], t.current["alive"]
    pid = env.get_player_id()
    if alive[pid] != USER_SEAT:
        raise HTTPException(400, "ユーザーの手番ではありません")
    state = env.get_state(pid)
    return make_hint("six_max", state)


@app.post("/tournaments/{tournament_id}/next")
def tournament_next(tournament_id: str):
    if tournament_id not in tournaments:
        raise HTTPException(404, "tournament not found")
    rec = tournaments[tournament_id]
    t = rec["t"]
    if t.finished:
        raise HTTPException(400, "トーナメントは終了しています")
    if t.current is not None:
        raise HTTPException(400, "ハンドがまだ終わっていません")
    t_start_hand(rec)
    return t_view(tournament_id)


if __name__ == "__main__":
    # `python api_server.py` でも起動できるようにする
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
