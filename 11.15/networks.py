import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.common_layer1 = nn.Linear(state_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.common_layer2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.output_layer = nn.Linear(512, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.norm1(self.common_layer1(x)))
        x = torch.relu(self.norm2(self.common_layer2(x)))
        return torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.hidden_layer1 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.hidden_layer2 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        self.output_layer = nn.Linear(256, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.norm1(self.input_layer(x)))
        x = torch.relu(self.norm2(self.hidden_layer1(x)))
        x = torch.relu(self.norm3(self.hidden_layer2(x)))
        return self.output_layer(x)
