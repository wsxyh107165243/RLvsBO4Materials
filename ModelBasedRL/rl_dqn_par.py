import os
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
from utils import collect_random, retry_on_error

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_size, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_size)

        self.reset_parameters()

        self.lr = q_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

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

class DQNAgent:
    def __init__(self, env: Environment = Environment()):
        self.env           = env
        self.state_dim     = self.env.state_dim
        self.action_size   = self.env.act_dim
        self.lr            = 5e-4                       # 5e-4, 1e-3
        self.gamma         = 1.00
        self.epsilon       = DEFAULT_INIT_EPSILON       # 0.95
        self.epsilon_decay = 0.99                       # TODO not sure 0.99 slow/fast enough, or should I choose another decay scheme?
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
            with torch.no_grad():
                self.Q.eval()
                state_tensor = torch.tensor(state.repr()).float().unsqueeze(0)
                q = self.Q(state_tensor)
                action = torch.argmax(q).item()
                maxQ_action_count = 1
                self.Q.train()
        else:
            action = self.env.sample_action()

        return action, None, maxQ_action_count
    
    def get_state_qs(self, state: State) -> np.ndarray:
        with torch.no_grad():
            self.Q.eval()
            state_tensor = torch.tensor(state.repr()).float().unsqueeze(0)
            qs = self.Q(state_tensor).detach().numpy().flatten()
            self.Q.train()
        return qs

    def choose_action_greedy(self, state: State) -> int:
        with torch.no_grad():
            self.Q.eval()
            state_tensor = torch.tensor(state.repr()).float().unsqueeze(0)
            q = self.Q(state_tensor)
            action = torch.argmax(q).item()
            self.Q.train()
        return action

    def train_agent(self, ep):
        ''' ep for target update '''
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = self.memory.sample()
        a_batch = a_batch.type(torch.int64)

        with torch.no_grad():
            Q_prime_actions = self.Q(s_prime_batch).argmax(1).unsqueeze(1)
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

    return func(state.x)

def propose_candidates_to_exp(agent: DQNAgent, rand_act_prob: float, candidates_n: int):
    '''
        Propose candidates_n xs to do experiment.
    '''
    try_counter, try_max_n = 0, 500
    prop_x_key_set = set()
    prop_x_a_key_set = set()
    while len(prop_x_key_set) < candidates_n:
        state = agent.env.reset()
        done = False
        while not done:
            while True:
                if random.random() < rand_act_prob:
                    action = agent.env.sample_action()
                else:
                    qs = agent.get_state_qs(state)
                    sorted_idxs = np.argsort(qs)[::-1]
                    found_unused_action = False
                    for action in sorted_idxs:
                        tmp_key = State.encode_key(list(state.x) + [state.ep_no, action])
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

'''
    When{ 
        train_ep_n = 1
        rand_act_prob = agent.epsilon <current> with no reset
        prop_smpls_per_round = ~D or n * D
        collect_random() only at start
        and plus something else...
    }, the following strategy will gradually degenerate into <nearly> using RL with no GPR.
'''
# TODO: train_ep_n -> 50 or 100
@retry_on_error(max_retries = 3)
def rl_dqn_serial(func_name, 
                  func_dim, 
                  train_ep_n = 25, 
                  prop_round_n = 200, 
                  prop_smpls_per_round = 10, 
                  rand_act_prob = 0.25, 
                  id = str(uuid.uuid4())[:8], 
                  verbose = False,
                  suppress_warning = True):
    ''' Careful when using warning suppression! '''
    if suppress_warning:
        warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated.*")

    env = Environment(func_name = func_name, dim = func_dim)
    agent = DQNAgent(env)

    TRAIN_START_MIN_MEMORY = 500
    
    for prop_iter in range(prop_round_n):
        ''' 
            Although TRAIN_START_MIN_MEMORY random samples are collected, the
            immediate rewards are lazily evaluated. This is reasonable as we can
            start training, using random samples online.
        '''
        agent.memory = ReplayBuffer(20000, 128, 'cpu', env)
        agent.epsilon = DEFAULT_INIT_EPSILON

        collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY, env.all_actions)

        for EP in range(train_ep_n + 1):
            _ = state = env.reset()
            done = False
            maxQ_action_count = 0

            while not done:
                action, _, count = agent.choose_action(state)
                state_prime, reward, done = env.step(state, action)
                agent.memory.add(state, action, reward, state_prime, done)

                maxQ_action_count += count
                state = state_prime
                    
                '''
                    Train agent with a batch of buffered sars' samples.
                    EP to determine if time to target update.
                '''
                agent.train_agent(EP)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

        '''
            Propose new experiment x, update internal gpr surrogate.
        '''
        prop_x_buffer = propose_candidates_to_exp(agent, rand_act_prob, prop_smpls_per_round)
        env.update_surrogate_buffer(prop_x_buffer)
        env.update_surrogate()

        print('ID: {}, Prop.R: {:3d}, Exp.N: {:4d}, Bsf: {:.5f}'.format(
            id, prop_iter, agent.env.get_exp_number(), agent.env.get_best_score()
        ))
        # print(f'Prop. round {prop_iter}, total exp. samples: {agent.env.get_exp_number()}, best-so-far: {agent.env.get_best_score()}')

        if abs(round(agent.env.get_best_score(), 4) - 0.0) < 1e-4:
            break
        
    return [agent.env.func(_x) for _x in agent.env.surrogate_buffer_list]   # in experimental order

if __name__ == '__main__':
    par_N = 96                      # number of parallel experiments, 48, 96, or 192
    func_name = 'ackley'            # name for test functions, ackley, rastrigin
    func_dim = 5                    # dim for test functions, 3 - 10
    train_ep_n = 200                # number of training episodes for RL agent, 100, 200
    prop_round_n = 200              # number of experimental prop. rounds, 
    prop_smpls_per_round = 4        # number of prop. samples per round, 1, 2, 4, 8, 16, total prop. samples = prop_round_n * prop_smpls_per_round
    rand_act_prob = 0.1             # 0.05, 0.1, 0.2

    for prop_smpls_per_round in [1, 2, 4, 8, 16]:
        id = str(uuid.uuid4())[:8]
        # prop_round_n = 25 * func_dim    # for higher dimensionality, more experimental round are needed.
        par_res = joblib.Parallel(n_jobs = -1)(joblib.delayed(rl_dqn_serial)(
                func_name, func_dim, train_ep_n, prop_round_n, prop_smpls_per_round, rand_act_prob, id
            ) for _ in range(par_N))
        joblib.dump(par_res, f'{func_name}-{func_dim}-{train_ep_n}-{prop_round_n}-{prop_smpls_per_round}-{rand_act_prob}-{id}.pkl')