'''
    RL environment with a GPR as an internally maintained surrogate.
    The immediate rewards are from the surrogate. Only after the RL
    proposition will the surrogate be updated to capture the current
    relationship between the experimented xs and ys.
'''
import math
from typing import List
from copy import deepcopy
import uuid
import warnings
from bayes_opt import UtilityFunction
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
from dqn.utils import ackley, rastrigin, levy, griewank, func_params

ROUND_DIGIT = 3
STATE_DELIMETER_CHAR = '*'

class State:
    def __init__(self, x, ep_no: float, action_idx: int, all_actions: List[float]):
        D = len(x)
        self.x = deepcopy(x)
        self.ep_no = ep_no
        self.action_idx = action_idx
        
        if self.ep_no >= 0 and self.ep_no < D and (action_idx is not None):
            self.x[self.ep_no] = all_actions[self.action_idx]
        else:
            assert self.ep_no < D
            assert action_idx is None
        
        self.x = np.array(self.x).round(ROUND_DIGIT)

        self.key: str = State.encode_key(self.x.tolist())

    def done(self):
        return self.ep_no == len(self.x) - 1
    
    def repr(self):
        return self.x.tolist() + [self.ep_no]
    
    def repr_bk(self):
        ''' reduced repr without ep_no is not good '''
        return self.x.tolist()
    
    @staticmethod
    def encode_key(x):
        return STATE_DELIMETER_CHAR.join(map(str, x))

    @staticmethod
    def decode_key(key: str):
        return np.array(list(map(float, key.split(STATE_DELIMETER_CHAR))))
    
class Environment:
    def __init__(self, 
                 func_name = 'ackley', 
                 dim = 4, 
                 init_N = 150, 
                 init_random_seed = None,
                 enable_ei: bool = False):
        assert func_name in func_params, f'{func_name} function not supported'
        self.dim = dim
        self.func_name = func_name
        self.x_min = func_params[func_name]['x_min']
        self.x_max = func_params[func_name]['x_max']
        self.act_dim = func_params[func_name]['act_dim']
        self.min_func = func_params[func_name]['func']
        self.func = lambda x: -self.min_func(x)  # minimization -> maximization
        
        self.all_actions = np.linspace(self.x_min, self.x_max, self.act_dim).round(ROUND_DIGIT).tolist()
        self.state_dim = len(self.reset().repr())

        self.best_score = float('inf')
        self.best_x = None

        self._random_state = 42
        self.surrogate = None
        self.surrogate_buffer = set()   # stores experimented xs
        self.surrogate_buffer_list = []
        self.cached_surrogate_pred_dict = dict()

        self.init_N = init_N
        self.enable_ei = enable_ei
        
        self.init_random_seed = init_random_seed
        self.init_surrogate(self.init_N)

    def init_surrogate(self, init_N):
        ''' initialize surrogate with init_N randomly generated samples '''
        __counter, __max_acc_count = 0, int(1e4)
        np_random_state = ensure_rng(self.init_random_seed)
        while len(self.surrogate_buffer) < init_N:
            _x = [np_random_state.choice(self.all_actions) for _i in range(self.dim)]
            _x_key = State.encode_key(_x)
            if _x_key not in self.surrogate_buffer:
                self.surrogate_buffer.add(_x_key)
                self.surrogate_buffer_list.append(_x)

            # loop out with error
            __counter += 1
            if __counter >= __max_acc_count:
                assert False, 'Potential permanent forloop!'

        ''' train a GPR model with the latest experimented xs, ys '''
        self.update_surrogate()

    def update_surrogate(self):
        ''' update surrogate '''
        if len(self.surrogate_buffer) == 0:
            return
        
        train_x = [State.decode_key(_x_key) for _x_key in self.surrogate_buffer]
        train_y = [self.func(_x) for _x in train_x]

        self._update_surrogate(train_x, train_y)

    # def pseudo_update_surrogate(self, k = 1):
    #     ''' pseudo update surrogate '''
    #     total_len = len(self.surrogate_buffer)
    #     train_x_real = self.surrogate_buffer_list[:total_len - k]
    #     train_y_real = [self.func(_x) for _x in train_x_real]

    #     self._update_surrogate(train_x_real, train_y_real)

    #     ''' fake update '''
    #     train_x_fake = self.surrogate_buffer_list[total_len - k:]
    #     train_y_fake = [self.surrogate_predict(_x) for _x in train_x_fake]

    #     train_x = train_x_real + train_x_fake
    #     train_y = train_y_real + train_y_fake
        
    #     self._update_surrogate(train_x, train_y)

    def _update_surrogate(self, train_x, train_y):
        ''' update surrogate '''
        train_x, train_y = np.array(train_x), np.array(train_y)
        # update self.best_score
        _best_idx = np.argmax(train_y.reshape(-1))
        self.best_score = train_y.reshape(-1)[_best_idx]
        self.best_x = train_x[_best_idx]

        ''' sklearn gpr '''
        self.surrogate = GaussianProcessRegressor(
            kernel = Matern(nu=2.5),
            alpha = 1e-6,
            normalize_y = True,
            n_restarts_optimizer = 5,
            random_state = self._random_state,
        )
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate.fit(train_x, train_y)

        if self.enable_ei:
            self.ei_acqf = UtilityFunction(kind = "ei", xi = 0.0)

        ''' reset cached prediction '''
        self.cached_surrogate_pred_dict = dict()

    def surrogate_predict(self, x):
        _x_key = State.encode_key(x)
        if _x_key not in self.cached_surrogate_pred_dict:
            x = np.atleast_2d(x).reshape(1, -1)
            if not self.enable_ei:
                pred_val = self.surrogate.predict(x)[0]
            else:
                pred_val = _ei = self.ei_acqf.utility(x, gp = self.surrogate, y_max = self.best_score)
            self.cached_surrogate_pred_dict[_x_key] = pred_val
            return pred_val
        else:
            return self.cached_surrogate_pred_dict[_x_key]

    def update_surrogate_buffer(self, sample_xs):
        ''' update experimented xs and ys '''
        sample_xs = np.array(sample_xs)
        for _x in sample_xs:
            _x_key = State.encode_key(_x)
            # assert _x_key not in self.surrogate_buffer, \
            #     f'Logic leak, using duplicate xs that are already in the surrogate_buffer: {_x.tolist()}'
            self.surrogate_buffer.add(_x_key)
            self.surrogate_buffer_list.append(_x)

    def check_collided(self, sample_x):
        _x_key = State.encode_key(sample_x)
        return _x_key in self.surrogate_buffer

    def reset(self):
        ''' 
            For RL, using a fixed initial point to start exploration is reasonable. 
            Think of it as an empty chess board.
        '''
        _idx = 15
        return State([self.all_actions[_idx] for _ in range(self.dim)], -1, None, self.all_actions)
    
    def step(self, state: State, action_idx: int):
        next_state = State(state.x, state.ep_no + 1, action_idx, self.all_actions)

        '''
            Using internally maintained GPR model to calculate reward.

            NOTE: Current implementation uses immediately calculated im.R.
            TODO: Use lazily calculated im.R if time complexity scales up.
        '''
        curr_score, next_score = self.surrogate_predict(state.x), self.surrogate_predict(next_state.x)
        reward = (next_score - curr_score)
        done = next_state.done()

        return next_state, reward, done
    
    def sample_action(self):
        return np.random.randint(0, self.act_dim)
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_x(self):
        return self.best_x
    
    def get_exp_number(self):
        assert len(self.surrogate_buffer) == len(self.surrogate_buffer_list), \
            'Something went wrong, len(surrogate_buffer) != len(surrogate_buffer_list)' + \
            f'{len(self.surrogate_buffer)}, {len(self.surrogate_buffer_list)}'
        return len(self.surrogate_buffer) - self.init_N