import torch
import torch.nn.functional as F
import numpy as np
import random

from DDPG.memory import ReplayBuffer
from DDPG.networks import Actor, Critic
from DDPG.noise import OrnsteinUhlenbeckActionNoise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, actor_lr, critic_lr, state_dims, action_dims, tau=0.001, gamma=0.99, buffer_size=1000000, fc1_size=400, fc2_size=300, batch_size=64, seed=0):
        random.seed(seed)

        self.actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(device)
        self.critic = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(device)
        self.target_actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(device)
        self.target_critic = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(device)

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
        torch.save(self.actor.state_dict(), "data/Actor.ddpg")
        torch.save(self.critic.state_dict(), "data/Critic.ddpg")
        torch.save(self.target_actor.state_dict(), "data/TargetActor.ddpg")
        torch.save(self.target_critic.state_dict(), "data/TargetCritic.ddpg")

    def load(self):
        self.actor.load_state_dict(torch.load("data/Actor.ddpg"))
        self.critic.load_state_dict(torch.load("data/Critic.ddpg"))
        self.target_actor.load_state_dict(torch.load("data/TargetActor.ddpg"))
        self.target_critic.load_state_dict(torch.load("data/TargetCritic.ddpg"))
