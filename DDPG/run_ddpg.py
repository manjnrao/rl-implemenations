import gym
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from DDPG.agent import Agent


def solve(agent, env, max_episodes, goal):
    start_time = time.time()
    scores = list()
    writer = SummaryWriter("tb_data")

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
    env.seed(0)
    agent = Agent(actor_lr=0.0001, critic_lr=0.001, state_dims=8, action_dims=2)

    solve(agent, env, 2000, 200)
