import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import shap  # Import SHAP library
from V2XEnvironment import V2XEnvironment  # Ensure correct import of V2XEnvironment

# Choose device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network Definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Common layers
        self.common_layer1 = nn.Linear(state_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.common_layer2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)

        # Output layer
        self.output_layer = nn.Linear(256, action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero

    def forward(self, state):
        x = torch.relu(self.norm1(self.common_layer1(state)))
        x = torch.relu(self.norm2(self.common_layer2(x)))
        action = torch.tanh(self.output_layer(x))  # Use tanh to keep actions within [-1, 1]
        return action

# Critic Network Definition
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Input layers
        self.input_layer = nn.Linear(state_dim + action_dim, 512)  # Increased layer size for more complexity
        self.norm1 = nn.LayerNorm(512)
        self.hidden_layer1 = nn.Linear(512, 512)  # Added extra hidden layer for increased depth
        self.norm2 = nn.LayerNorm(512)
        self.hidden_layer2 = nn.Linear(512, 256)  # Modified existing hidden layer
        self.norm3 = nn.LayerNorm(256)
        self.output_layer = nn.Linear(256, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# DDPG Agent Definition
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim

        # Initialize Actor networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=5e-6, weight_decay=1e-4)

        # Initialize Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=5e-6, weight_decay=1e-4)

        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = 256
        self.discount = 0.99
        self.tau = 0.002

        # Exploration noise parameters
        self.max_action = 1.0
        self.noise_std = 0.1


        # Initialize SHAP explainer (optional)
        self.explainer = None  # Initialize when needed

    def select_action(self, state, exploration=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if exploration:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0  # Not enough experiences to train

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)

        # Critic update - Train multiple times to improve Q value approximation
        for _ in range(3):  # Increase the number of training iterations for the Critic
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                next_action = torch.clamp(next_action, -self.max_action, self.max_action)
                target_q = self.critic_target(next_state, next_action)
                target_q = reward + (1 - done) * self.discount * target_q

            current_q = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

        # Actor update
        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)

    # Initialize SHAP explainer using background data
    def initialize_shap_explainer(self, background_states):
        actor_wrapper = ActorWrapper(self.actor)
        self.explainer = shap.GradientExplainer(agent.actor, background_states)

# Actor Wrapper for SHAP
class ActorWrapper(nn.Module):
    def __init__(self, actor_model):
        super(ActorWrapper, self).__init__()
        self.actor = actor_model

    def forward(self, state):
        action = self.actor(state)
        return action

# Training the Agent
env = V2XEnvironment()  # Create the environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim)  # Initialize the agent

# Initialize SHAP explainer with background states
background_states = []
for _ in range(100):
    state = env.reset()
    background_states.append(state)
background_states = torch.FloatTensor(np.array(background_states)).to(device)
agent.initialize_shap_explainer(background_states)

num_episodes = 3000
all_rewards = []
actor_losses = []
critic_losses = []

for episode in range(num_episodes):
    agent.noise_std = max(0.1 * (0.995 ** episode), 0.01)
    state = env.reset()
    episode_reward = 0+1
    done = False
    step = 0

    while not done:
        action = agent.select_action(state)

        # Apply action to the environment
        next_state, reward, done, _ = env.step(action)

        # Store transition in replay buffer
        agent.add_to_replay_buffer((state, action, reward, next_state, float(done)))

        # Train the agent
        actor_loss, critic_loss = agent.train()

        state = next_state
        episode_reward += reward
        step += 1

        # Optionally compute SHAP values
        if agent.explainer is not None and np.random.rand() < 0.01:
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            shap_values = agent.explainer.shap_values(state_tensor)

            # Process SHAP values as needed

    all_rewards.append(episode_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    # Plot results every 1000 episodes
    if (episode + 1) % 500 == 0:
        plt.figure()
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG on V2X Environment')
        plt.show()

        plt.figure()
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses')
        plt.legend()
        plt.show()
