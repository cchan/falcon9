import gym
from gym import spaces
from gym.utils import EzPickle
import random


# Dummy env to validate my training code against.
# Each observation is a random number. To get a reward, the next action must be that number.
class DummyEnv(gym.Env, EzPickle):
    def __init__(self):
        EzPickle.__init__(self)
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self):
        self.correct = None
        self.state = random.randint(0, 3)
        self.steps_taken = 0
        return self.state

    def step(self, action):
        self.correct = (action == self.state)
        if self.correct:
            reward = +1
        else:
            reward = -1
        self.state = random.randint(0, 3)
        self.steps_taken += 1
        if self.steps_taken >= 10:
            return self.state, reward, True, None
        else:
            return self.state, reward, False, None

    def render(self, mode='human'):
        print(self.correct, self.state)

    def close(self):
        pass
