"""
train_dqn.py - Phase 3: DQNボットの学習
ランダム相手に1万ハンド学習し、1000ハンドごとにルールボットと腕試しする
"""
import torch
import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import get_device, reorganize, tournament
from my_agents import RuleBasedAgent

device = get_device()  # GPUがあれば自動で使う(なければCPU)

# 学習用と評価用で別の環境を用意する
env = rlcard.make('no-limit-holdem', config={'seed': 42})
eval_env = rlcard.make('no-limit-holdem', config={'seed': 43})

agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64, 64],   # 脳のサイズ: 64ニューロン x 2層
    device=device,
)

env.set_agents([agent, RandomAgent(num_actions=env.num_actions)])
eval_env.set_agents([agent, RuleBasedAgent()])

NUM_EPISODES = 10000
print(f"学習開始: {NUM_EPISODES}ハンド (数分かかります)")
print("-" * 50)

for episode in range(1, NUM_EPISODES + 1):
    # 1ハンドプレイして経験を記憶に蓄積 → 定期的に脳を更新
    trajectories, payoffs = env.run(is_training=True)
    trajectories = reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
        agent.feed(ts)

    # 1000ハンドごとにルールボットと500ハンドの真剣勝負
    if episode % 1000 == 0:
        reward = tournament(eval_env, 500)[0]
        print(f"  {episode:>5}ハンド学習済 | vs ルールボット: {reward*100:+7.1f} BB/100")

torch.save(agent, 'dqn_model.pth')
print("-" * 50)
print("学習完了! モデルを dqn_model.pth に保存しました")
