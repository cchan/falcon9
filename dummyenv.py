import gym
from gym import spaces
from gym.utils import EzPickle
import numpy as np


# EchoEnv
# Dummy env to validate my training code against.
# Goal is to simply repeat the first observed number, with L1 error based reward.
class EchoEnv(gym.Env, EzPickle):
    STEP_LIMIT = 10000

    def __init__(self):
        EzPickle.__init__(self)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([5, 5]))
        self.action_space = spaces.Discrete(5)
        self.reset()

    def reset(self):
        self.squared_error = None
        self.state = self.observation_space.sample()
        self.steps_taken = 0
        return self.state

    def step(self, action):
        self.squared_error = (self.state[0].item() - action)**2
        self.state = self.observation_space.sample()
        self.steps_taken += 1
        if self.steps_taken > self.STEP_LIMIT:
            raise EnvironmentError("Episode is over.")
        # I don't think the +10 matters, but I want to give it a positive reward anyway. :)
        reward = 10 - self.squared_error
        return self.state, reward, self.steps_taken == self.STEP_LIMIT, None

    def render(self, mode='human'):
        print("Squared Error:", self.squared_error, "State:", self.state)

    def close(self):
        pass
