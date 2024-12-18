from time import sleep
import os
from typing import Tuple
import uuid
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import warnings

from environment import *
from buffer import ReplayBuffer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_size, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 256)
        self.fc_2 = nn.Linear(256, 1024)
        self.fc_out = nn.Linear(1024, action_size)

        self.reset_parameters()

        self.lr = q_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # TODO: dropout, elu & attention

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        q = F.relu(self.fc_1(x))
        q = F.relu(self.fc_2(q))
        q = self.fc_out(q)
        return q

DEFAULT_INIT_EPSILON = 0.95
DEFAULT_END_EPSILON = 0.01
DEFUALT_EPSILON_DECAY = 0.99

# TODO parameterize default values
class DQNAgent:
    def __init__(self, env: Environment = Environment()):
        self.env           = env
        self.state_dim     = self.env.state_dim
        self.action_size   = self.env.act_dim
        self.lr            = 1e-3                       # 5e-4, 1e-3
        self.gamma         = GAMMA
        self.epsilon       = DEFAULT_INIT_EPSILON       # 0.95
        self.epsilon_decay = DEFUALT_EPSILON_DECAY      # TODO not sure 0.99 slow/fast enough, or should I choose another decay scheme?
        self.epsilon_min   = DEFAULT_END_EPSILON
        self.targ_update_n = 10
        self.test_every    = 100
        self.memory        = ReplayBuffer(20000, 128, 'cpu', self.env)      # NOTE perhaps too large?

        self.Q        = QNetwork(self.state_dim, self.action_size, self.lr)
        self.Q_target = QNetwork(self.state_dim, self.action_size, self.lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state: State):
        random_number = np.random.rand()
        maxQ_action_count = 0
        if self.epsilon < random_number:
            action = self.choose_action_greedy(state)
            maxQ_action_count = 1
        else:
            action = self.env.sample_action(state)

        return action, None, maxQ_action_count
    
    def choose_action_greedy(self, state: State) -> int:
        qs, ai_low, ai_high = self.get_state_qs(state)
        action = np.argmax(qs[ai_low: ai_high + 1]) + ai_low
        return action
    
    def get_state_qs(self, state: State) -> Tuple[np.ndarray, int, int]:
        with torch.no_grad():
            self.Q.eval()
            state_tensor = torch.tensor(state.repr()).float().unsqueeze(0)
            qs = self.Q(state_tensor).detach().numpy().flatten()
            self.Q.train()
        act_idx_low, act_idx_high = state.get_action_idx_limits()
        return qs, act_idx_low, act_idx_high

    def train_agent(self, ep):
        ''' ep for target update '''
        s_batch, a_batch, r_batch, s_prime_batch, done_batch, next_act_idx_bounds = self.memory.sample()
        a_batch = a_batch.type(torch.int64)

        with torch.no_grad():
            # Q_prime_actions = self.Q(s_prime_batch).argmax(1).unsqueeze(1)
            # TODO checking 
            Q_prime_qs_list = self.Q(s_prime_batch)
            Q_prime_actions = [(Q_prime_qs[ai_low: ai_high + 1]).argmax().item() + ai_low \
                               for Q_prime_qs, (ai_low, ai_high) in zip(Q_prime_qs_list, next_act_idx_bounds)]
            Q_prime_actions = torch.tensor(Q_prime_actions).reshape(-1, 1)
            Q_target_next = self.Q_target(s_prime_batch).gather(1, Q_prime_actions)
            Q_targets = r_batch + self.gamma * (1 - done_batch) * Q_target_next

        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.mse_loss(Q_a, Q_targets)     # NOTE: or smooth_l1_loss?
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()

        ''' Important to clip the grads to avoid exploding gradients '''
        for param in self.Q.parameters():
            param.grad.data.clamp_(min = -1., max = 1.)
        
        self.Q.optimizer.step()

        if ep % self.targ_update_n == 0: 
            for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                param_target.data.copy_(param.data)

    def save_model(self, path):
        torch.save({
            'Q_state_dict': self.Q.state_dict(),
            'Q_target_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.Q.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
        self.Q.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def test_agent(agent: DQNAgent, func):
    state = agent.env.reset()
    done = False
    while not done:
        action = agent.choose_action_greedy(state)
        state_prime, _, done = agent.env.step(state, action)
        state = state_prime

    return func(state.get_composition())

# TODO check
def propose_candidates_to_exp(agent: DQNAgent, ei_agent: DQNAgent, ei_act_prob: float, candidates_n: int):
    '''
        Propose candidates_n xs to do experiment.
    '''
    assert 0. <= ei_act_prob <= 1.

    try_counter, try_max_n = 0, 500
    prop_x_key_set = set()
    prop_x_a_key_set = set()
    while len(prop_x_key_set) < candidates_n:
        state = agent.env.reset()
        done = False
        while not done:
            while True:
                # TODO check modify returned val of get_state_qs()
                if random.random() < ei_act_prob:
                    qs, ai_low, ai_high = ei_agent.get_state_qs(state)
                else:
                    qs, ai_low, ai_high = agent.get_state_qs(state)
                sorted_idxs = np.argsort(qs[ai_low: ai_high + 1])[::-1] + ai_low    # sort a slice
                found_unused_action = False
                for action in sorted_idxs:
                    tmp_key = State.encode_key(list(state.x) + [state.get_episode_count(), action])
                    if tmp_key not in prop_x_a_key_set:
                        found_unused_action = True
                        prop_x_a_key_set.add(tmp_key)
                        break
                if found_unused_action:
                    break
            state_prime, _, done = agent.env.step(state, action)
            state = state_prime
        new_x = state.x
        new_x_key = State.encode_key(new_x)
        if (not agent.env.check_collided(new_x)) and (new_x_key not in prop_x_key_set):
            prop_x_key_set.add(new_x_key)
        
        try_counter += 1
        if try_counter >= try_max_n: 
            raise Exception(f'Tried {try_counter} times to propose candidates. Reached maximum allowed attempts.')
    return [State.decode_key(_x_key) for _x_key in prop_x_key_set]

def train_one_ep(agent: DQNAgent, env: Environment, EP: int):
    ''' train <agent> on <env>, with EP_ID of <EP> '''
    _ = state = env.reset()
    done = False

    while not done:
        action, _, _ = agent.choose_action(state)
        state_prime, reward, done = env.step(state, action)
        agent.memory.add(state, action, reward, state_prime, done)

        state = state_prime
            
        '''
            Train agent with a batch of buffered sars' samples.
            EP to determine if time to target update.
        '''
        agent.train_agent(EP)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

def retry_on_error(max_retries = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    retries += 1
            raise Exception(f"Function {func.__name__} failed after {max_retries} retries.")
        return wrapper
    return decorator

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = state.generate_random_action()
        action_idx = ACTIONS_TO_INDEX_DICT[action]
        next_state, reward, done = env.step(state, action_idx)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    print(f'Collected {num_samples} random samples')