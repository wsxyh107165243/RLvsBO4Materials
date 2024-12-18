#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The main script for running a single replication.
"""
import errno
import json
import os
import sys
import functools
from time import sleep
from typing import Any, Dict
import warnings

import joblib
import numpy as np
import torch

from joblib import Parallel, delayed
from discrete_mixed_bo.experiment_utils import Rastrigin

from discrete_mixed_bo.run_one_replication import run_one_replication

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.rl_dqn_par import rl_dqn_serial

def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs

def bo_job(n_init, bo_iter, seed = 0):
    # additional args in config file
    dim = 10
    _args = ['__place_holder__', 'discrete_rastrigin', 'cont_optim__round_after__ei', seed, dim]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, _args[1])
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    
    ''' 
        Modify the configuration
        Also stick to the original implementation details from Samuel Daulton.
            kwargs['batch_size'] == 1
            ......
    '''
    kwargs['n_initial_points'] = n_init
    kwargs['iterations'] = bo_iter
    
    label = _args[2]
    seed = int(float(_args[3]))
    # dim = _args[4]
    _tr_label = '__tr' if 'use_trust_region' in kwargs and kwargs['use_trust_region'] else ''
    output_path = os.path.join(exp_dir, f'compounded_dim_{dim}_' + label + _tr_label, f"{str(seed).zfill(4)}_{label}.pt")

    ''' prepare output directory '''
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    save_callback = lambda data: torch.save(data, output_path)
    save_frequency = 5
    fetch_data(kwargs=kwargs)

    ''' run the BO loop, return the experimented X and Y '''
    X, Y = run_one_replication(
        seed=seed,
        label=label,
        dim=dim,
        save_callback=save_callback,
        save_frequency=save_frequency,
        **kwargs,
    )
    x_min, x_max = -5., 5.
    X = np.array(X) * (x_max - x_min) + x_min
    return X, Y

def rl_job(init_X, prop_round_n,seed = 0):
    func_name = 'rastrigin'
    dim = 10
    train_ep_n = 200
    prop_smpls_per_round = 1
    rand_act_prob = 0.1
    env_init_n = -1

    return rl_dqn_serial(func_name, dim, env_init_n, seed, init_X, train_ep_n, prop_round_n, prop_smpls_per_round, rand_act_prob)

def serial_job(seed = 0, 
               n_init = 100, 
               bo_iter = 300,
               rl_iter = 0):
    init_X, __init_Y = bo_job(n_init, bo_iter, seed)
    exp_X = rl_job(init_X, rl_iter, seed)
    exp_X = np.array(exp_X)
    np.save(f'exp_X_rastrigin_10_compounded_{n_init}_{bo_iter}_{rl_iter}_{seed}.npy', exp_X)

if __name__ == "__main__":
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 20, 80, 300) for seed in range(144))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 100, 300, 0) for seed in range(144))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 100, 0, 300) for seed in range(144))
    # serial_job(0, 20, 10, 0)