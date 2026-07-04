"""
inspect_state.py - AIが見ている「状態」の正体を暴く
1ハンドをステップ実行し、各判断時点の情報を人間語で表示する
"""
import rlcard

env = rlcard.make('no-limit-holdem', config={'seed': 7})

# env.run() を使わず、手動で1ステップずつ進める
state, player_id = env.reset()

step = 0
while not env.is_over():
    step += 1
    raw = state['raw_obs']  # 人間が読める形式の情報

    print("=" * 50)
    print(f"[Step {step}] プレイヤー{player_id} の番")
    print(f"  自分のハンド : {raw['hand']}")
    print(f"  ボード       : {raw['public_cards']}")
    print(f"  ポット       : {raw['pot']}")
    print(f"  各自のベット : {raw['all_chips']}")
    print(f"  残りスタック : {raw['stakes']}")
    print(f"  ステージ     : {raw['stage']}")
    print(f"  合法な行動   : {state['raw_legal_actions']}")

    # とりあえず「合法手の中から最初のもの」を選ぶ(=超単純なボット)
    action = state['legal_actions'].keys().__iter__().__next__()
    print(f"  → 選んだ行動: {state['raw_legal_actions'][0]}")

    state, player_id = env.step(action)

print("=" * 50)
print(f"ハンド終了! 収支: {env.get_payoffs()} (単位: BB)")
