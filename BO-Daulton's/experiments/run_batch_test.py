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

from dqn.rl_dqn_par import rl_dqn_serial

def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs

def bo_job(n_init, bo_iter, batch_size, seed = 0):
    # additional args in config file
    dim = 6
    _args = ['__place_holder__', 'discrete_ackley', 'cont_optim__round_after__ei', seed, dim]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, _args[1])
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    
    ''' 
        Modify the configuration manually.
        Also stick to the original implementation details from Samuel Daulton.
    '''
    kwargs['n_initial_points'] = n_init
    kwargs['iterations'] = bo_iter
    kwargs['batch_size'] = batch_size
    
    label = _args[2]
    seed = int(float(_args[3]))

    _tr_label = '__tr' if 'use_trust_region' in kwargs and kwargs['use_trust_region'] else ''
    output_path = os.path.join(exp_dir, f'init_{n_init}_q_{batch_size}_' + label + _tr_label, f"{str(seed).zfill(4)}_{label}.pt")

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
    return run_one_replication(
        seed=seed,
        label=label,
        dim=dim,
        save_callback=save_callback,
        save_frequency=save_frequency,
        **kwargs,
    ), output_path

def serial_job(seed = 0, 
               n_init = 100, 
               bo_iter = 300,
               batch_size = 1):
    (X, Y), outpath = bo_job(n_init, bo_iter, batch_size, seed)
    outdir = os.path.dirname(outpath)
    X = np.array(X)
    np.save(os.path.join(outdir, f'{n_init}_{bo_iter}_{batch_size}_{seed}.npy'), X)

if __name__ == "__main__":
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 10, 290, 1) for seed in range(96))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 30, 270, 1) for seed in range(96))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 50, 250, 1) for seed in range(96))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 70, 230, 1) for seed in range(96))
    Parallel(n_jobs = 48)(delayed(serial_job)(seed, 90, 210, 1) for seed in range(96))