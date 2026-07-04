# API仕様書 (フロントエンド開発用)

プレイ画面(play.html / サイト本番)はこの仕様に対して実装すること。
バックエンド実装状況: ✅=実装済み 🚧=v6で追加予定

## 共通
- ベースURL: `http://127.0.0.1:8000`
- アクションID: 0=fold, 1=check_call, 2=raise_half_pot, 3=raise_pot, 4=all_in
- カード表記: スート+ランク。例 `"SA"`=♠A, `"HT"`=♥10, `"D2"`=♦2, `"C9"`=♣9

## トーナメント(6人卓)

### ✅ POST /tournaments
テーブル作成。body: `{"level": "beginner|intermediate|advanced", "starting_stack": 200, "blind_up_every": 10}`

### ✅ GET /tournaments/{id} / POST /tournaments/{id}/action / POST /tournaments/{id}/next
レスポンス共通形:
```json
{
  "tournament_id": "abc123",
  "level": "intermediate",
  "seats": {"1": {"name": "嵐山レン", "style": "ルース・アグレッシブ (LAG)", "description": "..."}},
  "standings": {"hand_no": 3, "big_blind": 2, "stacks": {"0": 200, "1": 150},
                 "alive": [0,1,3], "finished": false, "winner": null},
  "hand_actions": [{"seat": 1, "action_id": 3}],
  "hand_finished": false,
  "last_hand": {"hand_no": 2, "big_blind": 2, "payoffs_game_units": [...], "seats_in_hand": [...]},
  "my_turn": true,
  "hand": ["SA", "HK"],
  "public_cards": ["D7", "C2", "HQ"],
  "pot": 12.0,
  "stage": "FLOP",
  "bets": {"0": 4, "1": 4, "3": 0},
  "legal_actions": [{"id": 0, "name": "fold"}, {"id": 1, "name": "check_call"}]
}
```

### ✅ GET /tournaments/{id}/hint
```json
{"recommended_action": {"id": 0, "name": "fold"},
 "reason": "あなたの 75o はティア7。GTOの参加レンジ外です...",
 "source": "GTOプリフロップ表"}
```

### 🚧 v6追加フィールド(既存レスポンスに追加される)
```json
{
  "win_probability": 0.34,          // 現時点の勝率(モンテカルロ推定)。手番時のみ
  "showdown": {                      // ショーダウン到達時のみ(hand_finished=true とセット)
    "hands": {"1": ["SA","SK"], "3": ["D9","D8"]},
    "rank_names": {"1": "フラッシュ", "3": "ワンペア"},
    "winners": [1]
  },
  "ai_hands_revealed": {"1": ["SA","SK"]}   // ユーザーがフォールドしたハンドの終了時、
                                             // 残ったAIのハンドを開示(学習用)
}
```

### 🚧 GET /tournaments/{id}/analysis (即時解説)
直前に終了したハンドの振り返り。ハンド終了後のみ有効。
```json
{
  "hand_no": 3,
  "your_actions": [
    {"stage": "PREFLOP", "action": "check_call",
     "gto_action_probs": {"fold": 0.7, "check_call": 0.3},   // NFSP平均方針による近似
     "ev_loss_estimate": 1.2,                                  // Q値差による近似(BB)
     "verdict": "Blunder"}                                     // Excellent/OK/Mistake/Blunder
  ],
  "summary_text": "プリフロップの75oコールがこのハンド最大のミスです。...(3行解説)",
  "summary_source": "llm"   // llm=Claude生成 / template=定型文(オフライン時フォールバック)
}
```

## ヘッズアップ(1対1) — ✅ /tables 系(同構造の簡易版)
POST /tables, GET /tables/{id}, POST /tables/{id}/action, GET /tables/{id}/hint, POST /tables/{id}/next
v6追加フィールドはトーナメントと共通仕様。

## フロント実装上の注意
- AIのアクションは1レスポンスにまとめて返る(`hand_actions`の差分を順次表示する演出はフロント側で)
- `bets` はそのストリートで各席が出しているチップ。フォールド判定は `hand_actions` の action_id=0 から
- 数値はすべて実チップ単位(ブラインド上昇はサーバー側で換算済み)
- エラーは `{"detail": "メッセージ"}` + 4xx
