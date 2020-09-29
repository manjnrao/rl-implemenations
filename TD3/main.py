import gym

from TD3.agent import Agent
from TD3.trainer import Trainer

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    agent = Agent(env, env.observation_space.shape, env.action_space.shape, actor_lr=0.001, critic_lr=0.001, seed=9)
    # agent.load()
    trainer = Trainer("TD3", agent, env)
    trainer.solve_env(goal=200, max_episodes=1000)
    # trainer.play()
