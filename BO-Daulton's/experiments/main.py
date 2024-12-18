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

import numpy as np
import torch

from joblib import Parallel, delayed
from discrete_mixed_bo.experiment_utils import Rastrigin

from discrete_mixed_bo.run_one_replication import run_one_replication

def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs

def serial_job(seed = 0):
    '''
        Additional args in .config file.
        _args : [place_holder, func_name, acq_name, seed, dim]
    '''
    _args = ['__place_holder__', 'discrete_rastrigin', 'cont_optim__round_after__ei', seed, 4]


    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, _args[1])
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    label = _args[2]
    seed = int(float(_args[3]))
    dim = _args[4]
    _tr_label = '__tr' if 'use_trust_region' in kwargs and kwargs['use_trust_region'] else ''
    output_path = os.path.join(exp_dir, f'dim_{dim}_' + label + _tr_label, f"{str(seed).zfill(4)}_{label}.pt")

    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    save_callback = lambda data: torch.save(data, output_path)
    save_frequency = 5
    fetch_data(kwargs=kwargs)

    run_one_replication(
        seed=seed,
        label=label,
        dim=dim,
        save_callback=save_callback,
        save_frequency=save_frequency,
        **kwargs,
    )

if __name__ == "__main__":
    Parallel(n_jobs = 8)(delayed(serial_job)(seed) for seed in range(96))