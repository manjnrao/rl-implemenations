import gym
from DDPG.trainer import Trainer


if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')

    trainer = Trainer("DDPG", env, seed=0)
    trainer.solve_env(goal=200, max_episodes=1000)
