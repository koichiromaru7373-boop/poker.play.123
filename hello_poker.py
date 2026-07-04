import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('no-limit-holdem', config={'seed': 42})

print("=" * 40)
print("RLCard 動作確認テスト")
print("=" * 40)
print(f"プレイヤー数     : {env.num_players}")
print(f"行動の種類数     : {env.num_actions}")
print(f"状態の形状       : {env.state_shape}")

agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)

trajectories, payoffs = env.run(is_training=False)

print("-" * 40)
print("1ハンド完了!")
for i, payoff in enumerate(payoffs):
    result = "勝ち" if payoff > 0 else ("負け" if payoff < 0 else "引き分け")
    print(f"プレイヤー{i}: {payoff:+.1f} BB ({result})")
print("=" * 40)
print("環境構築成功です。Phase 1 に進めます。")
