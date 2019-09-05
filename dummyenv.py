import gym
from gym import spaces
from gym.utils import EzPickle
import random
import numpy as np


# EchoEnv
# Dummy env to validate my training code against.
# Goal is to simply repeat the observed numbers, with MSE-based reward.
# Slightly ill-conditioned box, intentionally.
class EchoEnv(gym.Env, EzPickle):
    def __init__(self):
        EzPickle.__init__(self)
        self.observation_space = self.action_space = spaces.Box(
            low=np.array([-0.1, -1, -10]), high=np.array([0.1, 1, 10]))

        self.reset()

    def reset(self):
        self.mse = None
        self.state = self.observation_space.sample()
        self.steps_taken = 0
        return self.state

    def step(self, action):
        self.mse = np.square(self.state - action).mean()
        if self.mse > 10000:
            print(self.state, action, self.mse)
        self.state = self.observation_space.sample()
        self.steps_taken += 1
        # I don't think the +10 matters, but I want to give it a positive reward anyway. :)
        reward = 10 - self.mse
        if self.steps_taken > 10000:
            raise EnvironmentError("Episode is over.")
        elif self.steps_taken == 10000:
            return self.state, reward, True, None
        else:
            return self.state, reward, False, None

    def render(self, mode='human'):
        print("MSE:", self.mse, "State:", self.state)

    def close(self):
        pass
