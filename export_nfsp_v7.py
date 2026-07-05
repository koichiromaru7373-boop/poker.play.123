r"""v7(NFSP)のベストを、デプロイ用の軽量モデルとして models/ に書き出す。

NFSPは以下を内部に抱えていてファイルが巨大になる:
- RLエージェント(DQN)のリプレイメモリ
- 平均方策学習用のリザーババッファ
推論(eval_step, 平均方策)にはどちらも不要なので空にして保存する。

    .\venv\Scripts\python.exe export_nfsp_v7.py

出力: models/heads_up_advanced_v7.pth
    → api_server.py の advanced(heads_up) がこれを最優先で読む。
"""

import os

import torch

SRC = os.path.join("experiments", "nfsp_v7", "nfsp_model_best.pth")
DST = os.path.join("models", "heads_up_advanced_v7.pth")


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"見つからない: {SRC}(学習が best を保存済みか確認)")

    agent = load(SRC)
    # nfsp_model_best.pth は agents[0](単体) を保存している想定。
    # 万一 list(nfsp_last 形式) を渡された場合は先頭を採用。
    if isinstance(agent, (list, tuple)):
        agent = agent[0]

    before = os.path.getsize(SRC) / 1e6

    # --- RL(DQN)側のリプレイメモリを空に ---
    rl = getattr(agent, "_rl_agent", None)
    if rl is not None and hasattr(rl, "memory"):
        try:
            rl.memory.memory = []          # rlcard Memory の中身
        except Exception:
            pass
        for attr in ("batch_size", "memory_size"):
            pass  # 構造は保持。中身だけ捨てる

    # --- SL側のリザーババッファを空に ---
    rb = getattr(agent, "_reservoir_buffer", None)
    if rb is not None:
        for attr in ("_data",):
            if hasattr(rb, attr):
                try:
                    setattr(rb, attr, [])
                except Exception:
                    pass
        if hasattr(rb, "_add_calls"):
            try:
                rb._add_calls = 0
            except Exception:
                pass

    # --- 評価は平均方策(均衡に近い)で固定 ---
    if hasattr(agent, "evaluate_with"):
        try:
            agent.evaluate_with = "average_policy"
        except Exception:
            pass

    os.makedirs("models", exist_ok=True)
    torch.save(agent, DST)
    after = os.path.getsize(DST) / 1e6

    print(f"OK: {SRC} ({before:.1f}MB) -> {DST} ({after:.1f}MB)")
    print("次: git add models/heads_up_advanced_v7.pth して commit/push")
    print("※ 事前に『slim版でも eval_step が動くか』を必ずローカルで1ハンド確認すること")


if __name__ == "__main__":
    main()
