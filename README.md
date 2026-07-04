# ポーカーAI テストプレイ

RLCard製のポーカーAI(ノーリミットホールデム)と対戦できるWebアプリです。
1対1(ヘッズアップ)と、個性の違うAI5人との6人トーナメントが遊べます。

## 遊べるもの
- **6人トーナメント**: あなた+AI5人。ブラインドが10ハンドごとに倍になり、最後の1人まで
  - AIには性格があります(超攻撃型のレン、堅実なミサキ、コール魔のゴンさん、岩のイワオ、学習型AIのゼロ)
- **1対1**: 学習済みAIとのヘッズアップ(APIドキュメント画面から)
- **学習機能**: AIの推奨アクションと理由(ヒント)、推定勝率の表示、ハンド終了時の全ハンド開示

## 動かし方(Windows)

前提: [Python 3.10以上](https://www.python.org/downloads/)(インストール時に "Add Python to PATH" にチェック)

1. このリポジトリをダウンロード(緑の Code ボタン → Download ZIP → 展開)
2. `run_server.bat` をダブルクリック(初回は環境構築で数分かかります)
3. ブラウザで http://127.0.0.1:8000/play を開く

### Mac / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api_server:app --port 8000
```

### スマホから遊ぶ
サーバーを `uvicorn api_server:app --host 0.0.0.0` で起動し、
同じWi-Fiのスマホから `http://<PCのIPアドレス>:8000/play` を開く。

## 学習済みモデルについて(任意)
モデルファイル(`experiments/**/*.pth`)はサイズの都合でリポジトリに含まれていません。
**無くてもルールベースAIで遊べます**が、強いAIと遊ぶには配布された `experiments`
フォルダをプロジェクト直下に置いてください(Releases または開発者から入手)。

| 置き場所 | 効果 |
|---|---|
| `experiments/dqn_v5/dqn_model_best.pth` | 1対1の上級AIが学習モデルになる |
| `experiments/dqn_v2/dqn_model_best.pth` | 1対1の中級AIが学習モデルになる |
| `experiments/dqn_6max/dqn6_model_best.pth` | トーナメントの「ゼロ」が学習モデルになる |

## 開発者向け
- API仕様: [API_SPEC.md](API_SPEC.md)
- プロジェクト全体の文脈・学習スクリプトの使い方: [CLAUDE.md](CLAUDE.md)
- 学習の実行例: `python train_dqn_v5.py --num_episodes 30000`(詳細はCLAUDE.md)

## 注意
- 認証なし・テーブル状態はメモリ保持(サーバー再起動で消えます)。**テスト用途限定**で、
  インターネットへの直接公開はしないでください(LAN内または各自ローカル実行を推奨)
- 不具合報告は Issue へ。「どの画面で・何をしたら・どうなったか」を書いてもらえると助かります
