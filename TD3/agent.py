import torch
import numpy as np
from TD3.networks import Critic, Actor
from common.memory import ReplayBuffer
import torch.nn.functional as F


class Agent:
    def __init__(self, env, state_dims, action_dims, actor_lr, critic_lr, tau=0.005, gamma=0.99, buffer_size=1000000,
                 fc1_size=400, fc2_size=300, batch_size=100, noise=0.1, update_frequency=2, warmup=1000, seed=0):
        np.random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)
        self.critic_1 = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)
        self.critic_2 = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)

        self.target_actor = Actor(
            actor_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)
        self.target_critic_1 = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)
        self.target_critic_2 = Critic(
            critic_lr, state_dims, action_dims, fc1_size, fc2_size, seed=seed).to(self.device)

        self.memory = ReplayBuffer(buffer_size, seed=seed)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.learning_step_counter = 0
        self.warmup = warmup
        self.time_step = 0
        self.update_frequency = update_frequency

        self.soft_update(init=True)

    def soft_update(self, init=False):
        tau = 1 if init else self.tau

        actor_params = dict(self.actor.named_parameters())
        critic1_params = dict(self.critic_1.named_parameters())
        critic2_params = dict(self.critic_2.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic1_params = dict(self.target_critic_1.named_parameters())
        target_critic2_params = dict(self.target_critic_1.named_parameters())

        for param in critic1_params:
            critic1_params[param] = tau * critic1_params[param].clone() + (1-tau) * target_critic1_params[param].clone()

        for param in critic2_params:
            critic2_params[param] = tau * critic2_params[param].clone() + (1-tau) * target_critic2_params[param].clone()

        for param in actor_params:
            actor_params[param] = tau * actor_params[param].clone() + (1 - tau) * target_actor_params[param].clone()

        self.target_actor.load_state_dict(actor_params)
        self.target_critic_1.load_state_dict(critic1_params)
        self.target_critic_2.load_state_dict(critic2_params)

    def save(self):
        torch.save(self.actor.state_dict(), "data/Actor.td3")
        torch.save(self.critic_1.state_dict(), "data/Critic1.td3")
        torch.save(self.critic_2.state_dict(), "data/Critic2.td3")
        torch.save(self.target_actor.state_dict(), "data/TargetActor.td3")
        torch.save(self.target_critic_1.state_dict(), "data/TargetCritic1.td3")
        torch.save(self.target_critic_2.state_dict(), "data/TargetCritic2.td3")

    def load(self):
        self.actor.load_state_dict(torch.load("data/Actor.td3"))
        self.critic_1.load_state_dict(torch.load("data/Critic1.td3"))
        self.critic_2.load_state_dict(torch.load("data/Critic2.td3"))
        self.target_actor.load_state_dict(torch.load("data/TargetActor.td3"))
        self.target_critic_1.load_state_dict(torch.load("data/TargetCritic1.td3"))
        self.target_critic_2.load_state_dict(torch.load("data/TargetCritic2.td3"))

    def get_action(self, state):
        if self.time_step < self.warmup:
            action = torch.tensor(np.random.normal(scale=1, size=self.action_dims))
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            mu = self.actor.forward(state).to(self.device)
            action = mu + torch.tensor(np.random.normal(scale=self.noise), dtype=torch.float).to(self.device)

        action = torch.clamp(action, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return action.cpu().detach().numpy()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn()

    def sample_from_buffer(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_from_buffer()

        target_actions = self.target_actor.forward(next_states)
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_target = self.target_critic_1.forward(next_states, target_actions)
        q2_target = self.target_critic_2.forward(next_states, target_actions)

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)

        q1_target[dones] = 0
        q2_target[dones] = 0

        q_target = torch.min(q1_target, q2_target)
        target = rewards + self.gamma * q_target
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        loss = q1_loss + q2_loss
        loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learning_step_counter += 1

        if self.learning_step_counter % self.update_frequency != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.soft_update()
