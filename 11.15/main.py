import matplotlib.pyplot as plt
from agent import DDPGAgent
from V2XEnvrionment import V2XEnvironment
import torch

# 初始化环境和代理
env = V2XEnvironment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = DDPGAgent(state_dim, action_dim)

# 训练超参数
num_episodes = 5000
all_rewards = []
actor_losses = []
critic_losses = []

# 创建目录以保存模型和图表
import os
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        td_error = reward  # 初始TD-error，可以根据实际情况计算
        agent.add_experience(state, action, reward, next_state, done, td_error)

        # 训练代理
        actor_loss, critic_loss = agent.train()

        state = next_state
        episode_reward += reward

    # 保存每个 episode 的奖励和损失
    all_rewards.append(episode_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)

    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    # 每 500 个回合保存一次图表
    if (episode + 1) % 500 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards, label='Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG on V2X Environment - Rewards')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/rewards_episode_{episode + 1}.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/losses_episode_{episode + 1}.png')
        plt.close()

# 保存最终模型
torch.save(agent.actor.state_dict(), 'models/ddpg_actor_final.pth')
torch.save(agent.critic.state_dict(), 'models/ddpg_critic_final.pth')

# 绘制并保存最终的奖励图
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG on V2X Environment - Final Rewards')
plt.legend()
plt.tight_layout()
plt.savefig('plots/rewards_final.png')
plt.close()

# 绘制并保存最终的损失图
plt.figure(figsize=(10, 5))
plt.plot(actor_losses, label='Actor Loss')
plt.plot(critic_losses, label='Critic Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Actor and Critic Losses - Final')
plt.legend()
plt.tight_layout()
plt.savefig('plots/losses_final.png')
plt.close()
