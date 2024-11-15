import torch
import torch.optim as optim
from replay_buffer import PrioritizedReplayBuffer
from networks import Actor, Critic
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=500000, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size

        # 网络初始化
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        # 同步目标网络权重
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-6)

        # 优先经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

        # 超参数
        self.discount = 0.99
        self.tau = 0.005
        self.noise_std = 0.15  # 初始噪声标准差
        self.noise_std_min = 0.05  # 最小噪声标准差
        self.noise_decay = 1e-5 # 每步减小的量

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().detach().numpy().flatten()
        noise = np.random.normal(0, self.noise_std, size=self.action_dim)
        self.noise_std = max(self.noise_std - self.noise_decay, self.noise_std_min)  # 逐步减小噪声
        return np.clip(action + noise, -1, 1)

    def add_experience(self, state, action, reward, next_state, done, td_error):
        self.replay_buffer.add(td_error, (state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer.tree.data) < self.batch_size:
            return 0.0, 0.0

        # 从缓冲区采样数据
        batch, indices, is_weights = self.replay_buffer.sample(batch_size=self.batch_size)

        # 解包数据
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

        # 将 Python 列表转换为 NumPy 数组，然后再转换为 PyTorch 张量
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
        batch_next_state = np.array(batch_next_state)
        batch_done = np.array(batch_done)
        is_weights = np.array(is_weights)

        # 转换为 PyTorch 张量
        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_action = torch.FloatTensor(batch_action).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(device)

        # Critic 更新
        critic_loss_total = 0.0
        for _ in range(3):  # Critic 更新 3 次
            with torch.no_grad():
                next_action = self.target_actor(batch_next_state)
                target_q = self.target_critic(batch_next_state, next_action)
                target_q = batch_reward + (1 - batch_done) * self.discount * target_q

            current_q = self.critic(batch_state, batch_action)
            td_error = target_q - current_q
            critic_loss = (is_weights * td_error ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.4)  # 添加梯度裁剪
            self.critic_optimizer.step()

            critic_loss_total += critic_loss.item()

        critic_loss_avg = critic_loss_total / 3

        # Actor 更新
        actor_loss = -self.critic(batch_state, self.actor(batch_state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)  # 添加梯度裁剪
        self.actor_optimizer.step()

        # 更新优先级
        td_error_np = td_error.detach().cpu().numpy().squeeze()
        self.replay_buffer.update_priorities(indices, td_error_np)

        # 软更新目标网络
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return actor_loss.item(), critic_loss_avg

    def _soft_update(self, main_net, target_net):
        for main_param, target_param in zip(main_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)
