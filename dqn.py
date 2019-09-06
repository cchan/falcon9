# With help from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
# import gym
from itertools import count
from dummyenv import EchoEnv

EXPERIENCE_REPLAY_SIZE = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('prev_obs', 'action', 'obs', 'reward'))

# Memory for experience replay.
memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)


# The Deep Q-Learning function approximator, which is just a fully connected
# neural net with ReLU.
class FCN(nn.Module):
    # First (input) size should be the observation size.
    # Last (output) size should be the action space size.
    def __init__(self, sizes):
        super(FCN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(curr, next)
                                     for curr, next in zip(sizes, sizes[1:])])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


env = EchoEnv()

BATCH_SIZE = 128
GAMMA = 0.0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 100

# Get number of actions from gym action space
n_obs = env.observation_space.shape[0]  # posx, posy, velx, vely, angle, angleVel, leg1contact, leg2contact
n_actions = env.action_space.n  # ...

policy_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
print(policy_net)
target_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)


def epsilon_greedy(state, epsilon_greedy_annealing_step):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * epsilon_greedy_annealing_step / EPS_DECAY)

    if random.random() > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(torch.tensor(state))).item()
    else:
        return env.action_space.sample()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # Get a random batch to train on (experience replay)
    # sample should be better than choice (w/o replacement converges faster)
    transitions = random.sample(memory, BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*[torch.tensor(x) for x in zip(*transitions)])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    Q = policy_net(batch.prev_obs).gather(1, batch.action.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the
    # expected state value or 0 in case the state was final.
    # <-- I believe that this leader-follower thing is for the sake of training stability. -->
    Q_next_max = target_net(batch.obs).gather(1, batch.action.unsqueeze(1)).detach()
    # Compute the expected Q values

    Q_expected = batch.reward + GAMMA * Q_next_max

    #print(Q.shape, Q_next_max.shape, Q_expected.shape)

    # Compute Huber loss
    loss = F.smooth_l1_loss(Q, Q_expected)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


epsilon_greedy_annealing_step = 0
for episode in range(1000):
    prev_obs = env.reset()

    totalreward = 0

    for t in count():
        action = epsilon_greedy(prev_obs,
                                epsilon_greedy_annealing_step)
        epsilon_greedy_annealing_step += 1

        obs, reward, done, _ = env.step(action)
        totalreward += reward

        if len(memory) < 128:
            memory.append((prev_obs, action, obs, reward))
        prev_obs = obs

        optimize_model()

        if (t + 1) % 10 == 0:
            env.render()
        if done:
            print(f"Episode {episode} done after {t+1} steps, total reward {totalreward}")
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Done!")
env.close()
