# Gym: https://github.com/openai/gym/blob/master/docs/environments.md
# Torch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# TRFL: https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb https://medium.com/aureliantactics/basic-trfl-usage-ad2a7e1afdde

import torch
from collections import namedtuple
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EXPERIENCE_REPLAY_SIZE = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Memory for experience replay.
from collections import deque
memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)
# memory.append( ... )
# random.sample(memory, batch_size)
# sample should be better than choice, since without replacement usually converges faster.

# The Deep Q-Learning function approximator, which is just a fully connected neural net with ReLU, except sigmoid in the last layer.
class FCN(nn.Module):
    # First (input) size should be the observation size.
    # Last (output) size should be the action space size.
    def __init__(self, sizes):
        super(FCN, self).__init__()
        self.layers = [nn.Linear(curr, next) for curr, next in zip(sizes, sizes[1:])]
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return F.sigmoid(self.layers[-1](x))


import gym
env = gym.make('LunarLander-v2')

for episode in range(1000):
  print("Starting episode", episode)
  obs = env.reset()
  reward = done = info = None
  env.render()

  for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
      break

env.close()
