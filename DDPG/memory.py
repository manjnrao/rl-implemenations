import random
from collections import deque
import numpy as np


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
