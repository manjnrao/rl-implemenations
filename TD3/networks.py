import torch
from torch import nn, optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, seed=0):
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc_mu = nn.Linear(self.fc2_size, *self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        activation = self.fc1(state)
        activation = F.relu(activation)

        activation = self.fc2(activation)
        activation = F.relu(activation)

        activation = torch.tanh(self.fc_mu(activation))
        return activation


class Critic(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, seed=0):
        super(Critic, self).__init__()
        torch.manual_seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(self.state_dims[0] + self.action_dims[0], self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc_q = nn.Linear(self.fc2_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        q_value = self.fc1(torch.cat([state, action], dim=1))
        q_value = F.relu(q_value)

        q_value = self.fc2(q_value)
        q_value = F.relu(q_value)

        q_value = self.fc_q(q_value)
        return q_value
