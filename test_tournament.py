"""トーナメントラッパーの動作検証 (AIのみで1トーナメント自動進行)

実行:
    python test_tournament.py

確認ポイント:
- 最後の1人まで進んで優勝者が出るか
- チップ総量が一定に保たれるか(保存則)
- ブラインドが10ハンドごとに倍になるか
"""

import os

import torch

from my_agents import RuleBasedAgent
from tournament import Tournament

MODEL_6MAX = os.path.join("experiments", "dqn_6max", "dqn6_model_best.pth")


def load_or_rule(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    return RuleBasedAgent(num_actions=5)


if __name__ == "__main__":
    dqn6 = load_or_rule(MODEL_6MAX)
    agents = {
        0: dqn6,                            # 席0: 学習済み6maxモデル(ユーザー役)
        1: RuleBasedAgent(num_actions=5),
        2: RuleBasedAgent(num_actions=5),
        3: dqn6,
        4: RuleBasedAgent(num_actions=5),
        5: RuleBasedAgent(num_actions=5),
    }
    t = Tournament(agents, starting_stack=200, blind_up_every=10, seed=7)
    print(f"開始: 6人, 各200チップ, 10ハンドごとにブラインド倍\n")

    total0 = sum(t.stacks.values())
    while not t.finished and t.hand_no < 500:
        t.play_hand(verbose=True)
        total = sum(t.stacks.values())
        assert total == total0, f"チップ保存則違反: {total0} -> {total}"

    print(f"\n終了: {t.hand_no}ハンド, 優勝=席{t.winner}, 最終BB={t.big_blind}")
    if not t.finished:
        print("!! 500ハンドで決着せず(要調査)")
