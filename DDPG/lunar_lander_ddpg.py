import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gym
import time
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("tb_data")


# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None, seed=0):
        random.seed(seed)

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer:
    """ For Experience replay. """
    def __init__(self, memory_size=1000000, seed=0):
        random.seed(seed)

        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(tuple([state, action, reward, next_state, done]))

    def sample(self, batch_size=64):
        experiences = random.sample(self.memory, k=batch_size)

        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None])
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        dones = np.vstack([e[4] for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones


class Critic(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, name, checkpoint_dir='tmp', seed=0):
        super(Critic, self).__init__()
        random.seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg.ckp')

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

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, lr, state_dims, action_dims, fc1_size, fc2_size, name, checkpoint_dir='tmp', seed=0):
        super(Actor, self).__init__()
        random.seed(seed)

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_ddpg.ckp')

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

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, actor_lr, critic_lr, state_dims, action_dims, tau=0.001, gamma=0.99, buffer_size=1000000, fc1_size=400, fc2_size=300, batch_size=64, seed=0):
        random.seed(seed)

        self.actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, name="Actor", seed=seed).to(device)
        self.critic = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, name="Critic", seed=seed).to(device)
        self.target_actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, name="TargetActor", seed=seed).to(device)
        self.target_critic = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, name="TargetCritic", seed=seed).to(device)

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dims), seed=seed)

        self.memory = ReplayBuffer(buffer_size, seed=seed)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

        self.soft_update(init=True)

    def get_action(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).to(device)
        mu = self.actor(state).to(device)
        noise = torch.tensor(self.noise(), dtype=torch.float).to(device)
        action = mu + noise
        self.actor.train()

        return action.cpu().detach().numpy()

    def sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        return states, actions, rewards, next_states, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_from_buffer()

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(next_states)
        target_critic_value = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        target_critic_value[dones] = 0.0
        target = rewards + self.gamma * target_critic_value
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.soft_update()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn()

    def train_actor(self):
        pass

    def train_critic(self):
        pass

    def soft_update(self, init=False):
        tau = 1 if init else self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for param in critic_params:
            critic_params[param] = tau * critic_params[param].clone() + (1 - tau) * target_critic_params[param].clone()

        for param in actor_params:
            actor_params[param] = tau * actor_params[param].clone() + (1 - tau) * target_actor_params[param].clone()

        self.target_actor.load_state_dict(actor_params)
        self.target_critic.load_state_dict(critic_params)

    def save(self):
        self.actor.save()
        self.critic.save()
        self.target_critic.save()
        self.target_actor.save()

    def load(self):
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()


def solve(agent, env, max_episodes, goal):
    start_time = time.time()
    scores = list()

    for i_episode in range(1, max_episodes + 1):
        agent.noise.reset()
        state = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        mean_score = np.mean(scores[-100:])
        display_progress(i_episode, start_time, mean_score)
        writer.add_scalar("Score", score, i_episode)
        writer.flush()

        if i_episode % 20 == 0:
            agent.save()

        if mean_score > goal:
            print("Booyah!")
            break
    writer.close()


def display_progress(i_episode, start_time, score):
    h, m, s = get_hms(start_time)
    print('\r{:02d}:{:02d}:{:02d} - Episode {}\tAverage Score: {:.2f}'.format(
        h, m, s, i_episode, score), end="")

    if i_episode % 20 == 0:
        print('\r{:02d}:{:02d}:{:02d} - Episode {}\tAverage Score: {:.2f}'.format(
            h, m, s, i_episode, score))


def get_hms(start_time):
    seconds_passed = time.time() - start_time
    h = int(seconds_passed / 3600)
    m = int(seconds_passed / 60) % 60
    s = int(seconds_passed) % 60
    return h, m, s


if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(actor_lr=0.0001, critic_lr=0.001, state_dims=8, action_dims=2)

    solve(agent, env, 2000, 210)
