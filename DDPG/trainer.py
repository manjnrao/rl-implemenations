import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

from DDPG.agent import Agent


class Trainer:

    name = None
    env = None
    agent = None

    training_start_time = None
    training_duration = None
    training_scores = list()
    training_steps_taken = 0
    training_episodes_seen = 0

    def __init__(self, name, env, seed=0):
        self.name = name
        self.writer = SummaryWriter("tb_data")

        self.env = env
        self.env.seed(seed)

        self.agent = Agent(env.observation_space.shape, env.action_space.shape, actor_lr=0.0001, critic_lr=0.001, seed=seed)

    def play(self):
        # video_recorder = VideoRecorder(env, "play.mp4", enabled=True)

        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.get_action(state)
            self.env.render()
            # video_recorder.capture_frame()
            state, reward, done, _ = self.env.step(action)

        # video_recorder.close()
        self.env.close()

    def solve_env(self, goal, max_episodes):
        self.training_start_time = time.time()

        for i_episode in range(1, max_episodes+1):
            self.agent.noise.reset()
            state = self.env.reset()
            self.training_episodes_seen = i_episode

            score = 0
            done = False
            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                self.training_steps_taken += 1

                state = next_state
                score += reward
                if done:
                    break

            self.training_scores.append(score)
            mean_score = np.mean(self.training_scores[-100:])
            self.display_progress(i_episode, mean_score)
            self.writer.add_scalar("Score", score, i_episode)
            self.writer.flush()

            if i_episode % 20 == 0:
                self.save()

            if mean_score > goal:
                print("\n\nSolved in {} steps.".format(self.training_steps_taken))
                break

        self.writer.close()

    def save(self):
        self.agent.save()

    def load(self):
        self.agent.load()

    def display_progress(self, i_episode, mean_score):
        h, m, s = self.get_hms()
        print('\r{:02d}:{:02d}:{:02d} - Episode {}\tSteps {}\tAverage Score: {:.2f}'.format(
            h, m, s, i_episode, self.training_steps_taken, mean_score), end="")

        if i_episode % 20 == 0:
            print('\r{:02d}:{:02d}:{:02d} - Episode {}\tSteps {}\tAverage Score: {:.2f}'.format(
                h, m, s, i_episode, self.training_steps_taken, mean_score))

    def get_hms(self):
        seconds_passed = time.time() - self.training_start_time
        h = int(seconds_passed / 3600)
        m = int(seconds_passed / 60) % 60
        s = int(seconds_passed) % 60
        return h, m, s
