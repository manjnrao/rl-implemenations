import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, seed=0):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc_state_1 = nn.Linear(self.state_dims, self.fc1_size)
        fs1 = 1 / np.sqrt(self.fc_state_1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc_state_1.weight.data, -fs1, fs1)
        torch.nn.init.uniform_(self.fc_state_1.bias.data, -fs1, fs1)
        self.bn1 = nn.LayerNorm(self.fc1_size)

        self.fc_state_2 = nn.Linear(self.fc1_size, self.fc2_size)
        fs2 = 1 / np.sqrt(self.fc_state_2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc_state_2.weight.data, -fs2, fs2)
        torch.nn.init.uniform_(self.fc_state_2.bias.data, -fs2, fs2)
        self.bn2 = nn.LayerNorm(self.fc2_size)

        self.fc_action_1 = nn.Linear(self.action_dims, fc2_size)
        fa1 = 1 / np.sqrt(self.fc_action_1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc_action_1.weight.data, -fa1, fa1)
        torch.nn.init.uniform_(self.fc_action_1.bias.data, -fa1, fa1)

        self.fc_q = nn.Linear(self.fc2_size, 1)
        f = 0.003
        torch.nn.init.uniform_(self.fc_q.weight.data, -f, f)
        torch.nn.init.uniform_(self.fc_q.bias.data, -f, f)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

    def forward(self, state, action):
        state_activation = self.fc_state_1(state)
        state_activation = self.bn1(state_activation)
        state_activation = F.relu(state_activation)

        state_activation = self.fc_state_2(state_activation)
        state_activation = self.bn2(state_activation)
        # state_activation = F.relu(state_activation)

        action_activation = self.fc_action_1(action)
        # action_activation = F.relu(action_activation)

        q_activation = F.relu(torch.add(state_activation, action_activation))
        q_activation = self.fc_q(q_activation)
        return q_activation


class Actor(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, seed=0):
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(self.state_dims, self.fc1_size)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_size)

        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_size)

        self.fc_mu = nn.Linear(self.fc2_size, self.action_dims)
        f3 = 0.003
        torch.nn.init.uniform_(self.fc_mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc_mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        activation = self.fc1(state)
        activation = self.bn1(activation)
        activation = F.relu(activation)

        activation = self.fc2(activation)
        activation = self.bn2(activation)
        activation = F.relu(activation)

        activation = torch.tanh(self.fc_mu(activation))
        return activation
