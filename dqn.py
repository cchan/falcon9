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
from dummyenv import DummyEnv

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


env = DummyEnv()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_obs = 4  # posx, posy, velx, vely, angle, angleVel, leg1contact, leg2contact
n_actions = env.action_space.n

policy_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
target_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


def epsilon_greedy(state, epsilon_greedy_annealing_step):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * epsilon_greedy_annealing_step / EPS_DECAY)

    if random.random() > eps_threshold:
        with torch.no_grad():
            return int(policy_net(torch.tensor(state, dtype=torch.long)))
    else:
        return random.randrange(n_actions)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # Get a random batch to train on (experience replay)
    # sample should be better than choice (w/o replacement converges faster)
    transitions = random.sample(memory, BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.obs)), device=device,
                                  dtype=torch.uint8)
    non_final_next_states = torch.tensor([s for s in batch.obs
                                         if s is not None])
    state_batch = torch.tensor(batch.prev_obs)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the
    # expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states) \
        .max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # This will emit a warning about broadcasting, this is expected.
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


epsilon_greedy_annealing_step = 0
for episode in range(1000):
    prev_obs = env.reset()
    reward = done = info = None
    env.render()

    totalreward = 0

    for t in count():
        action = epsilon_greedy(prev_obs,
                                epsilon_greedy_annealing_step)
        epsilon_greedy_annealing_step += 1

        obs, reward, done, _ = env.step(action)
        totalreward += reward
        env.render()

        memory.append((prev_obs, action, obs, reward))
        prev_obs = obs

        print(memory)

        optimize_model()

        if done:
            print(f"Episode {episode} done after {t+1} steps, total reward {totalreward}")
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Done!")
env.close()
