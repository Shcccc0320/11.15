import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import os  # 导入 os 用于目录管理
from V2XEnvironment import V2XEnvironment  # 确保正确导入 V2XEnvironment

# 创建用于保存图表和模型的目录
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 选择设备：如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        # 公共层
        self.common_layer1 = nn.Linear(state_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.common_layer2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)

        # 输出层
        self.output_layer = nn.Linear(512, action_dim)

        # 初始化权重
        self.apply(self._init_actor_weights)

    def _init_actor_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 将偏置初始化为零

    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm1(self.common_layer1(input_state)))
        x = torch.relu(self.norm2(self.common_layer2(x)))
        action = torch.tanh(self.output_layer(x))  # 使用 tanh 保证动作在 [-1, 1] 范围内
        return action

# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(state_dim + action_dim, 512)  # 增加层的大小以提高复杂性
        self.norm1 = nn.LayerNorm(512)
        self.hidden_layer1 = nn.Linear(512, 512)  # 添加额外的隐藏层以增加深度
        self.norm2 = nn.LayerNorm(512)
        self.hidden_layer2 = nn.Linear(512, 256)  # 修改现有隐藏层
        self.norm3 = nn.LayerNorm(256)
        self.output_layer = nn.Linear(256, 1)

        # 初始化权重
        self.apply(self._init_critic_weights)

    def _init_critic_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_state: torch.Tensor, input_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_state, input_action], dim=1)
        x = self.input_layer(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.hidden_layer1(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.hidden_layer2(x)
        x = self.norm3(x)
        x = torch.relu(x)
        q_value = self.output_layer(x)
        return q_value

# DDPG 代理定义
class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim

        # 初始化 Actor 网络
        self.main_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.main_actor.state_dict())
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=1e-5)

        # 初始化 Critic 网络
        self.main_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.main_critic.state_dict())
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=5e-6)

        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = 256
        self.discount = 0.99
        self.tau = 0.005

        # 探索噪声参数
        self.initial_noise_std = 0.1
        self.noise_std = self.initial_noise_std
        self.noise_decay = 0.99995
        self.min_noise_std = 0.01

        self.max_action = 1.0
        self.current_episode = 0  # 当前回合数，用于更新噪声

    def select_action(self, current_state: np.ndarray, exploration: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(current_state.reshape(1, -1)).to(device)
        action = self.main_actor(state_tensor).cpu().data.numpy().flatten()

        if exploration:
            exploration_noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + exploration_noise
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train(self) -> tuple:
        # 初始化损失为0.0，确保在任何情况下都返回这两个变量
        actor_loss = 0.0
        critic_loss = 0.0

        if len(self.replay_buffer) < self.batch_size:
            return actor_loss, critic_loss  # 经验不足以训练

        try:
            batch = random.sample(self.replay_buffer, self.batch_size)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
            batch_action = torch.FloatTensor(np.array(batch_action)).to(device)
            batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
            batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
            batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

            # Critic 更新 - 多次训练以提高 Q 值估计
            critic_loss_total = 0.0
            for _ in range(3):  # 增加 Critic 的训练迭代次数
                with torch.no_grad():
                    next_action = self.target_actor(batch_next_state)
                    next_action = torch.clamp(next_action, -self.max_action, self.max_action)
                    target_q = self.target_critic(batch_next_state, next_action)
                    target_q = batch_reward + (1 - batch_done) * self.discount * target_q

                current_q = self.main_critic(batch_state, batch_action)
                loss = nn.MSELoss()(current_q, target_q)
                critic_loss_total += loss

                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.main_critic.parameters(), max_norm=0.4)
                self.critic_optimizer.step()

            critic_loss = (critic_loss_total / 3).item()  # 平均损失

            # Actor 更新
            actor_action = self.main_actor(batch_state)
            actor_loss_tensor = -self.main_critic(batch_state, actor_action).mean()
            actor_loss = actor_loss_tensor.item()

            self.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.main_actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # 更新目标网络
            for main_param, target_param in zip(self.main_critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)

            for main_param, target_param in zip(self.main_actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)

            return actor_loss, critic_loss
        except Exception as e:
            print(f"Error during training: {e}")
            return actor_loss, critic_loss  # 返回默认值以确保两个返回值

    def add_to_replay_buffer(self, transition: tuple):
        self.replay_buffer.append(transition)

    def update_noise(self):
        self.noise_std = max(self.initial_noise_std * (self.noise_decay ** self.current_episode), self.min_noise_std)

# 训练代理
def train_ddpg_agent():
    env = V2XEnvironment()  # 创建环境
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(state_dim, action_dim)  # 初始化代理

    num_episodes = 5000
    all_rewards = []
    actor_losses = []
    critic_losses = []

    for episode in range(num_episodes):
        agent.current_episode = episode  # 设置当前回合数
        agent.update_noise()  # 更新探索噪声
        current_state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(current_state)

            # 将动作应用于环境，并解包返回值
            try:
                next_state, reward, done, info = env.step(action)
            except Exception as e:
                print(f"Error during env.step at Episode {episode + 1}: {e}")
                next_state, reward, done = current_state, 0.0, True  # 设置默认值
                info = {}

            # 将转换存储在回放缓冲区中
            agent.add_to_replay_buffer((current_state, action, reward, next_state, float(done)))

            try:
                # 训练代理
                actor_loss, critic_loss = agent.train()
            except Exception as e:
                print(f"Training error at Episode {episode + 1}: {e}")
                actor_loss, critic_loss = 0.0, 0.0  # 设置默认值以避免未定义

            current_state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        # 每 500 个回合保存一次结果图
        if (episode + 1) % 500 == 0:
            # 绘制并保存奖励图
            plt.figure(figsize=(10, 5))
            plt.plot(all_rewards, label='Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('DDPG on V2X Environment - Rewards')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/rewards_episode_{episode + 1}.png')
            plt.close()

            # 绘制并保存损失图
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

    # 训练结束后，保存最终的模型和图表
    torch.save(agent.main_actor.state_dict(), 'models/ddpg_actor_final.pth')
    torch.save(agent.main_critic.state_dict(), 'models/ddpg_critic_final.pth')

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

if __name__ == "__main__":
    train_ddpg_agent()
