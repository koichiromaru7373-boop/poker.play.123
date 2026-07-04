"""デプロイ用の軽量モデルを書き出す

学習済みpthにはリプレイメモリ(数十MB)が含まれるため、推論に不要な部分を
除去して `models/` に保存する(数MB以下になり、Gitに同梱できる)。

実行:
    python export_model.py
"""

import os

import torch

EXPORTS = [
    # (元ファイル, 出力先)
    (os.path.join("experiments", "dqn_v5", "dqn_model_best.pth"),
     os.path.join("models", "heads_up_advanced.pth")),       # v5 (68次元/拡張env)
    (os.path.join("experiments", "dqn_v2", "dqn_model_best.pth"),
     os.path.join("models", "heads_up_intermediate.pth")),   # v2 (54次元/標準env)
    (os.path.join("experiments", "dqn_6max", "dqn6_model_best.pth"),
     os.path.join("models", "six_max.pth")),                 # 6max (54次元/標準env)
    (os.path.join("experiments", "nfsp_v6", "nfsp_model_best.pth"),
     os.path.join("models", "heads_up_v6.pth")),             # v6 (70次元/v6env) ※採用時のみ
]


def slim_export(src, dst):
    try:
        agent = torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        agent = torch.load(src, map_location="cpu")
    for attr in ("memory", "_reservoir_buffer"):  # 学習用バッファを除去
        if hasattr(agent, attr):
            setattr(agent, attr, None)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(agent, dst)
    print(f"{dst}: {os.path.getsize(dst) // 1024} KB (元 {os.path.getsize(src) // 1024 // 1024} MB)")


if __name__ == "__main__":
    for src, dst in EXPORTS:
        if os.path.exists(src):
            slim_export(src, dst)
        else:
            print(f"スキップ(なし): {src}")
