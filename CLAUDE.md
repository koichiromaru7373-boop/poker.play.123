# CLAUDE.md — ポーカーAI開発プロジェクト

## 目的
RLCardのno-limit-holdem環境でポーカーAIを開発し、FastAPIでAPI化して自社サイトの利用者と対戦できるようにする。

### プロダクト要件(自社サイト)
- 対戦はAIと利用者のみ(利用者同士の対戦はなし)
- 形式: 1対1(ヘッズアップ)と、利用者+AI5人の6人トーナメント
- 中級者/上級者ルーム(モデルの強さ差し替えで実現。初級はルールベースボットも可)
- 「打ち方を学べる」機能(AIの推奨アクションをヒントとして返す)
- ゲーム進行(ディーラー、ポット管理、役判定)はAPI側が全部持つ。サイトは表示と入力のみ

## 環境
- OS: Windows
- Python: 3.12(仮想環境 `venv` を使用。作業前に必ず activate すること: `venv\Scripts\activate`)
- rlcard 1.2.0 + setuptools + PyTorch(`rlcard[torch]`)導入済み

## 進捗
- [x] Phase 0: 環境構築
- [x] Phase 1: 状態(state/observation)の理解
- [x] Phase 2: ルールベースボット
  - `my_agents.py` に `RuleBasedAgent` を実装
  - 成績: ランダム相手に **+396 BB/100**
- [x] Phase 3: DQN学習
  - v2完了(50000ハンド・約20分): ベスト +15.57 BB/hand @30000、終盤は +600〜1000 BB/100 で推移
  - ※評価が500ハンドのためブレ大。「ベスト」は運の可能性あり
  - v3(継続学習・`--resume`対応)実行可能
- [ ] Phase 4: 評価・改善 ← いまここ
  - `evaluate.py` で保存済みモデルを10000ハンド以上で再評価しベストを確定する
  - `train_dqn_v4.py` 作成済み: 自己対戦+相手プール(RuleBased/Random/過去の自分)で過剰適合を防ぐ
  - 目標: 限りなく最強。v4で伸び止まったらNFSP(rlcard標準搭載)への移行を検討
- [ ] Phase 4.5: 6人テーブル用モデルの学習
  - `train_dqn_6max.py` 作成済み(未実行)。ヘッズアップ用と6人用の2系統のモデルを保持する
- [ ] Phase 5: FastAPIでAPI化、自社サイトと対戦
  - `api_server.py` 雛形作成済み: テーブル作成/状態取得/アクション/ヒント/次ハンド の5エンドポイント(1ハンド完結版)
  - TODO: チップ持ち越し・ブラインド上昇・脱落(トーナメント用ラッパー)、テーブルの永続化(現状インメモリ)、認証
  - 難易度: `api_server.py` の MODEL_PATHS でモデルファイル差し替え(初級=RuleBasedAgent)

## DQNの結論(2026-07-03)
- v4ベスト ≈ v3ベスト(直接対戦で差は誤差レベル)。DQN路線は頭打ちと判断
- 現時点の最強 = `experiments/dqn_v4/dqn_model_best.pth`(上級ルーム暫定モデル)
- さらなる強化は `train_nfsp.py`(NFSP自己対戦、dqn_v4ベストとの対戦評価付き)で継続中

## 主要ファイル
- `my_agents.py` — RuleBasedAgent(ルールベースボット)
- `train_dqn.py` — DQN学習スクリプト(Phase 3 v1)
- `train_dqn_v2.py` — v2: 50000ハンド、2500ごとに500ハンド評価、ベストを `dqn_model_best.pth` に保存、最後に成績一覧を表示
- `train_dqn_v3.py` — v3: デフォルト200000ハンド。`--resume <モデル.pth>` で継続、`--checkpoint_every`(既定10000)ごとに `dqn_model_last.pth` を保存
- `evaluate.py` — 保存済みモデルを大サンプル(既定10000ハンド)で再評価(複数モデル比較可)
- `train_dqn_v4.py` — v4: 自己対戦+相手プール。`--selfplay_ratio`(既定0.5)で過去の自分、0.1でRandom、残りRuleBasedと対戦。評価はvs Rule / vs Randomの平均でベスト判定
- `compare_models.py` — モデル同士の直接対戦(席入替あり)。自己対戦後の「vs Rule低下=劣化かバランス化か」の判定に使う
- `train_nfsp.py` — NFSP自己対戦(2体)。`--benchmark`(既定: dqn_v4ベスト)との対戦成績でベスト判定。デプロイ用 `nfsp_model_best.pth` と再開用 `nfsp_last.pth` を保存
- `train_dqn_6max.py` — 6人テーブル用DQN(v4と同じ相手プール方式)。出力は `experiments/dqn_6max/`
- `api_server.py` — FastAPI雛形(1ハンド完結)。`uvicorn api_server:app --reload` で起動、`pip install fastapi uvicorn` が必要。動作確認済み(ヘッズアップ・ヒント機能)
- `tournament.py` — トーナメント進行ラッパー(チップ持ち越し・ブラインド倍化・脱落・優勝判定)。ブラインド上昇は「実チップをブラインド単位換算してRLCardに渡す」方式。`decide_fn`で人間の手番を差し込める
- `test_tournament.py` — AIのみで1トーナメント自動進行し、チップ保存則・決着・ブラインド倍化を検証する
- `personas.py` — トーナメントの席1〜5のAIキャラ定義(LAG/TAG/ルースパッシブ/タイトパッシブ/学習型AI)。名前・スタイル・性格説明をAPIが返し、サイトで「人読み」練習に使う
- `my_agents.py` の `StyleAgent` — looseness(参加の広さ)と aggression(攻撃性)で性格を変えられるボット。ペルソナの中身
- `play.html` — テーブル囲み型の対戦画面。APIの `/play` で配信。楕円テーブル+6席、アクション吹き出し、ベット額表示、フォールド可視化(降りても観戦継続)、優勝オーバーレイ。サイト本番のプロトタイプ
- `gto_hints.py` — ヒント生成。プリフロップ=GTO定石のハンドティア表(説明文付き)、ポストフロップ=学習モデルの推奨+役分析の説明文。ヘッズアップのヒント役はNFSPベストがあれば自動で優先
- `gto_hints.py` の `GTOHybridAgent` — プリフロップ=GTO表・ポストフロップ=学習モデルのハイブリッド。APIの中級/上級AIは自動でこれにラップされる(初級RuleBasedと人格ボットは対象外)
- `extended_env.py` — 拡張観測環境(68次元: 標準54+ポットオッズ/スタック/ストリート/席)。`PreflopGTO`ラッパーも同居。**v5系モデルは旧モデルと互換なし・デプロイにもこのenvが必要**
- `train_dqn_v5.py` — ポストフロップ特化学習: 拡張観測+全員プリフロップGTO固定+ポストフロップ遷移のみ学習+相手プール。既定300000ハンド(数時間)、試運転は `--num_episodes 30000`。`--until 17:00` で時刻指定終了(`--num_episodes 10000000` と併用で時間いっぱい学習)
- `experiments/dqn_v2/`, `experiments/dqn_v3/` — 学習ログ(performance.csv)、学習曲線(fig.png)、モデルの出力先

## よく使うコマンド(Windows)
```
# v3: v2のベストから追加学習
.\venv\Scripts\python.exe train_dqn_v3.py --resume experiments\dqn_v2\dqn_model_best.pth --num_episodes 100000

# 再評価(ベストモデル確定用)
.\venv\Scripts\python.exe evaluate.py experiments\dqn_v2\dqn_model_best.pth experiments\dqn_v2\dqn_model_last.pth --num_games 10000
```

## 別PCでの学習
- `venv` はコピーしない。フォルダ一式(venv以外)をコピー → `setup_new_pc.bat` 実行で環境構築
- 学習の続きは `experiments/` 内の `*_last.pth` を持っていき `--resume` で再開
- 複数PCで並行する場合は役割分担する(例: PC1=NFSP、PC2=6max)。同じ学習を両方で回すと成果がマージできないため不可
- GPU(NVIDIA)搭載PCなら `pip install torch --index-url https://download.pytorch.org/whl/cu121` でCUDA版に差し替えると学習が数倍速い

## 開発体制
- 単独開発に変更(別Claudeとの分担は中止)。API_SPEC.md は仕様書として維持・更新する
- 実装済み: showdown開示 / フォールド時AIハンド開示 / 推定勝率(win_probability, MC300試行) — `equity.py`
- 実装済み: v5モデルのAPI載せ替え — ヘッズアップ上級は `dqn_v5/dqn_model_best.pth` があれば拡張観測envで自動採用(無ければv4)。ヒントも自卓AIを使用
- v5学習結果: 146万ハンド時点で vs GTO-Rule +2.4 BB/hand前後でプラトー(ポストフロップ専用値)
- エンジンv6作成済み: `extended_env_v6.py`(70次元: +equity/SPR、equityはMC60試行+キャッシュ)、`train_nfsp_v6.py`(NFSP自己対戦・プリフロップGTO固定・--until対応)。出力は `experiments/nfsp_v6/`
- play.html スマホ対応済み(640px以下でレスポンシブ)。スマホから遊ぶには `uvicorn api_server:app --host 0.0.0.0` で起動し、同じWi-FiからPCのIPで `http://<PCのIP>:8000/play`
- 残TODO(優先順): ①/analysis(即時解説) ②v6モデルのAPI載せ替え(extended_env_v6) ③6人卓のv5/v6化 ④永続化・認証

## 注意
- コマンド実行前に venv の activate を忘れない(または `.\venv\Scripts\python.exe` を直接使う)
- 評価指標は BB/100(100ハンドあたりのビッグブラインド収支)
- Cowork(Claude)のサンドボックスは外部ネットワーク制限によりPyPI不可・Windowsバイナリ実行不可。学習・評価の実行はローカルPCで行うこと
- Coworkのファイル同期でサンドボックス側が一時的に古い/途中までの内容に見えることがある。実体(Windows側)を基準に確認する
