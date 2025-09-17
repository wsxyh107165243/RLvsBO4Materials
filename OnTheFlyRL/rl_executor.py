import random
import uuid
import warnings

import joblib
import numpy as np

from buffer import ReplayBuffer
from environment import COMPOSITION_ROUNDUP_DIGITS, Environment
from rl_dqn_agents import (
    DQNAgent, 
    collect_random, propose_candidates_to_exp, train_one_ep, retry_on_error,
    DEFAULT_INIT_EPSILON,
)

'''
    When{ 
        train_ep_n = 1
        ei_act_prob = agent.epsilon <current> with no reset
        prop_smpls_per_round = ~D or n * D
        collect_random() only at start
        and plus something else...
    }, the following strategy will gradually degenerate into <nearly> using RL with no GPR.
'''
# @retry_on_error(max_retries = 1)
def rl_dqn_double_agents_serial(init_N = 20,
                                seed = 0,
                                train_ep_n = 150, 
                                prop_round_n = 200, 
                                prop_smpls_per_round = 10, 
                                ei_act_prob = 0.25, 
                                id = str(uuid.uuid4())[:8], 
                                suppress_warning = True):
    '''
        Train DQN agents using GPR rewards and the double agent scheme.
    '''
    ''' Careful when using warning suppression! '''
    if suppress_warning:
        warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated.*")

    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)
    agent = DQNAgent(env)

    ei_env = Environment(init_N = init_N, enable_ei = True, random_seed = seed)
    # NOTE: use the same num=init_N randomly initialized random points
    ei_env.copy_initialization(env) 
    ei_agent = DQNAgent(ei_env)

    TRAIN_START_MIN_MEMORY = 3000

    up_2_date_ei = []
    
    for prop_iter in range(prop_round_n):
        ''' 
            Although TRAIN_START_MIN_MEMORY random samples are collected, the
            immediate rewards are lazily evaluated. This is reasonable as we can
            start training, using random samples online.
        '''
        agent.memory = ReplayBuffer(20000, 128, 'cpu', env)
        agent.epsilon = DEFAULT_INIT_EPSILON
        collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY)

        ei_agent.memory = ReplayBuffer(20000, 128, 'cpu', ei_env)
        ei_agent.epsilon = DEFAULT_INIT_EPSILON
        collect_random(ei_env, ei_agent.memory, TRAIN_START_MIN_MEMORY)

        # TODO: debug to see if agent is well-trained
        ''' train mean agent '''
        for EP in range(train_ep_n + 1):
            train_one_ep(agent, env, EP)

        ''' train EI agent '''
        for EP in range(train_ep_n * 5 + 1):
            train_one_ep(ei_agent, ei_env, EP)

        ''' Propose new experiment x, update internal gpr surrogate. '''
        prop_x_buffer = propose_candidates_to_exp(agent, ei_agent, ei_act_prob, prop_smpls_per_round)
        env.update_surrogate_buffer(prop_x_buffer)
        env.update_surrogate()
        # print('updated mean agent')

        ei_env.update_surrogate_buffer(prop_x_buffer)
        ei_env.update_surrogate()
        # print('updated ei agent')

        ''' update up_2_date_ei '''
        up_2_date_ei.append(ei_env.surrogate_predict(prop_x_buffer[0]))

        print('ID: {}, Prop.R: {:3d}, Exp.N: {:4d}, Bsf: {:.5f}'.format(
            id, prop_iter, agent.env.get_exp_number(), agent.env.get_best_score()
        ))
        # print(f'Prop. round {prop_iter}, total exp. samples: {agent.env.get_exp_number()}, best-so-far: {agent.env.get_best_score()}')
        
    return (
        agent.env.surrogate_buffer_list,
        [agent.env.func(_x) for _x in agent.env.surrogate_buffer_list],   # in experimental order
        up_2_date_ei,
    )

def rl_dqn_serial(init_N = 20,
                  seed = 0,
                  train_ep_n = 150, 
                  ):
    '''
        Train one DQN agent using on-the-fly rewards.
    '''
    id_str = str(uuid.uuid4())[:8]

    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)

    ''' 
        Clear internal buffers.
        The internal buffers are intially built for GPR rewards.    TODO: 完善区分on-the-fly和surr reward的逻辑
    '''
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()

    agent = DQNAgent(env)

    TRAIN_START_MIN_MEMORY = 3000
    
    ''' 
        Although TRAIN_START_MIN_MEMORY random samples are collected, the
        immediate rewards are lazily evaluated. This is reasonable as we can
        start training, using random samples online.
    '''
    agent.memory = ReplayBuffer(3000, 256, 'cpu', env)
    agent.epsilon = DEFAULT_INIT_EPSILON
    collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY)

    traj = []

    ''' train ONE agent using on-the-fly rewards '''
    for ep in range(train_ep_n + 1):
        train_one_ep(agent, env, ep)
        if ep % 10 == 0:
            bsf = round(env.best_score, COMPOSITION_ROUNDUP_DIGITS)
            _tmp_res = [ep, len(env.surrogate_buffer), bsf]
            traj.append(_tmp_res)
            print(*_tmp_res)

    bsf_list = [env.func(_comp) for _comp in env.surrogate_buffer_list]

    bsf_list = np.maximum.accumulate(bsf_list).tolist()

    joblib.dump(bsf_list, f'rl_single_agent_direct_R-{id_str}.pkl')

if __name__ == '__main__':
    #rl_dqn_double_agents_serial()

    joblib.Parallel(n_jobs = 12)(
        joblib.delayed(rl_dqn_serial)(
            train_ep_n = 1200, seed = sd
        ) for sd in [random.randint(0, 999) for _ in range(24)])
