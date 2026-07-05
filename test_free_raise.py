"""free_raise の動作確認(ローカルの Windows/venv で実行)。

    .\venv\Scripts\python.exe test_free_raise.py

チェック内容:
- 任意額レイズを差し込んでもハンドが最後まで正常進行する
- チップ保存則(全員の in_chips + remained_chips が一定)が崩れない
- payoffs の合計が 0(ゼロサム)
- レンジ検証(小さすぎ/コール以下)が弾かれる
"""

import random

import rlcard

import free_raise
from free_raise import RaiseTo, raise_bounds, apply_raise_to

free_raise.enable()


def total_chips(env):
    return sum(int(p.in_chips) + int(p.remained_chips) for p in env.game.players)


def play_one_hand(num_players=3, seed=0, human_seat=0, verbose=False):
    random.seed(seed)
    env = rlcard.make("no-limit-holdem", config={"game_num_players": num_players, "seed": seed})
    env.reset()
    start_total = total_chips(env)

    steps = 0
    while not env.is_over():
        pid = env.get_player_id()
        state = env.get_state(pid)
        legal = list(state["legal_actions"].keys())
        legal_ids = [int(getattr(a, "value", a)) for a in legal]

        did_custom = False
        if pid == human_seat and any(i in legal_ids for i in (2, 3, 4)):
            b = raise_bounds(env)
            if b["max_to"] > b["my_in"] + b["to_call"]:
                # min と all-in の中間くらいを任意額でレイズ
                target = (b["min_to"] + b["max_to"]) // 2
                target = max(b["min_to"], min(target, b["max_to"]))
                before = total_chips(env)
                apply_raise_to(env, target)
                assert total_chips(env) == before, "レイズでチップ総量がずれた"
                if verbose:
                    print(f"  seat{pid} RAISE_TO {target} (added {target - b['my_in']})")
                did_custom = True

        if not did_custom:
            a = random.choice(legal_ids)
            env.step(a)
            if verbose:
                print(f"  seat{pid} action {a}")

        assert total_chips(env) == start_total, "チップ保存則が破れた"
        steps += 1
        assert steps < 500, "無限ループの疑い"

    payoffs = env.get_payoffs()
    assert abs(float(sum(payoffs))) < 1e-6, f"payoffs 非ゼロサム: {sum(payoffs)}"
    assert total_chips(env) == start_total, "終了後にチップ総量がずれた"
    return payoffs


def test_range_validation():
    env = rlcard.make("no-limit-holdem", config={"game_num_players": 3, "seed": 1})
    env.reset()
    b = raise_bounds(env)
    # コール額以下は弾かれる
    try:
        apply_raise_to(env, b["my_in"] + b["to_call"])
        raised = False
    except ValueError:
        raised = True
    assert raised, "コール以下のレイズが弾かれていない"


if __name__ == "__main__":
    print("=== range validation ===")
    test_range_validation()
    print("OK")

    print("=== play hands with custom raises ===")
    for seed in range(30):
        for n in (2, 3, 6):
            p = play_one_hand(num_players=n, seed=seed, verbose=(seed == 0 and n == 3))
    print("OK — 全ハンド正常進行・チップ保存・ゼロサム確認")
