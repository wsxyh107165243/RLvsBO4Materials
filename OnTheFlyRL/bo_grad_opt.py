from copy import deepcopy
import itertools
import math
from typing import Dict, List
import uuid
import warnings
import joblib
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import ensure_rng
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
)
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

from environment import (
    Environment, State, 
    COMP_LIMITS, ELEM_N, COMPOSITION_INTERVAL, EPISODE_LEN, COMP_MAX_LIMITS, COMP_MIN_LIMITS,
    get_ground_truth_func, get_mo_ground_truth_func
)

FLOAT_ROUND_DIGIT = 4

''' Just ignore numerous sklearn warnings '''
def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

class DiscreteCompositionBO(BayesianOptimization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def register_compositional_limits(self, 
                                      elem_bounds = COMP_LIMITS, 
                                      composition_interval = COMPOSITION_INTERVAL):
        """ Register compositional limits """  
        '''
            NOTE:   This is not a duplicate variable of self._space.bounds.
                    Will be used in acq_max func with the self-defined 
                    parameter order.
        '''
        self.elem_bounds = np.array(elem_bounds)
        self.elem_n = len(elem_bounds)
        self.comp_interval = composition_interval

        assert self.elem_n == ELEM_N
    
    def init_discrete_rand_samples(self, n_init_rand: int, seed: int, x_order_buff: List, bsf_buff: List):
        """ Initialize discrete random N points for BO to train a GPR """
        assert self.elem_bounds is not None, 'Compositional bounds is not initialized.'
        random_state = ensure_rng(seed)

        while len(self.space) < n_init_rand:
            _state = State(if_init = True)
            for _ in range(EPISODE_LEN):
                _state = State(previous_state = _state, action = _state.generate_random_action(random_state))
            _x = _state.get_composition()

            ''' NOTE params is stored in sorted keys order '''
            candidate_dis = self.space.array_to_params(_x)
            if self.contains(candidate_dis):
                continue
            target = self.space.target_func(**candidate_dis)
            self.register(params = candidate_dis, target = target)
        
            x_order_buff.append(_x)
            bsf_buff.append(self.max['target'])

    def _init_continuous_rand_samples(self, num: int) -> np.ndarray:
        """ Initialize continuous random N points as seeds for inner argmax of BO """
        random_state = self._random_state
        seeds_buffer = []
        for _seed_id in range(num):
            _seed = np.zeros(ELEM_N)
            for elem_index in range(len(_seed) - 1):
                _min = max(COMP_MIN_LIMITS[elem_index], 1 - sum(_seed[:elem_index]) - sum(COMP_MAX_LIMITS[elem_index + 1:]))
                _max = min(COMP_MAX_LIMITS[elem_index], 1 - sum(_seed[:elem_index]))
                _seed[elem_index] = random_state.uniform(_min, _max)
            _seed[-1] = 1.0 - _seed[:-1].sum()
            seeds_buffer.append(_seed)
        return np.array(seeds_buffer)
    
    def _init_continuous_rand_samples_rejection(self, num: int) -> np.ndarray:
        """ 
            Initialize continuous random N points as seeds for inner argmax of BO,
            using rejection sampling.
        """
        random_state = self._random_state
        seeds_buffer = []
        while len(seeds_buffer) < num:
            _tmp_comp = np.zeros(ELEM_N)
            _found_flag = True
            for i in range(ELEM_N - 1):
                _c = random_state.random() * (COMP_MAX_LIMITS[i] - COMP_MIN_LIMITS[i]) + COMP_MIN_LIMITS[i]
                if _tmp_comp.sum() + _c > 1.0:
                    _found_flag = False
                    break
                _tmp_comp[i] = _c
            _last_c = 1 - _tmp_comp.sum()
            if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
                _found_flag = False
            else:
                _tmp_comp[-1] = _last_c
            if _found_flag:
                seeds_buffer.append(_tmp_comp)
        return np.array(seeds_buffer)

    @ignore_warnings
    def suggest_contiuous_x(self, utility_function, constraints) -> List:
        """ Most promising point to probe next """
        if len(self._space) == 0:
            raise Exception('No initial samples')
            # return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        ac = utility_function.utility
        gp = self._gp
        y_max = self._space.target.max()
        bounds = self.elem_bounds
        random_state = self._random_state

        '''
            Number of times to run scipy.minimize. The default value in bayes_opt is 10.
        '''
        n_iter = 10

        ''' Main body of continuous_x inner loop (optimization) of BO '''
        continuous_x_buff = []
        
        # randomly sampled x_val seeds
        x_seeds = self._init_continuous_rand_samples_rejection(n_iter)
        
        # objective of scipy.optimize is minimization
        to_minimize = lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max)

        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: to_minimize(x),
                            x_try,
                            bounds = bounds,
                            method = 'SLSQP',
                            constraints = constraints)

            # See if success
            if not res.success:
                continue

            tmp_continuous_x = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            continuous_x_buff.append(tmp_continuous_x)

        return continuous_x_buff

    def continuous_to_discrete(self, candidate_continuous: List[float]) -> np.ndarray:
        """ Convert the continuous candidate to discrete candidates """
        assert len(candidate_continuous) == ELEM_N
        _usable_dis_f = []
        for elem_idx in range(ELEM_N):
            _x = candidate_continuous[elem_idx]
            _x_low_neighbor = math.floor(_x / self.comp_interval) * self.comp_interval
            _x_high_neighbor = math.ceil(_x / self.comp_interval) * self.comp_interval
            _dis_f = np.unique(np.clip(
                [_x_low_neighbor, _x_high_neighbor], 
                self.elem_bounds[elem_idx][0], 
                self.elem_bounds[elem_idx][1]
            ))
            _usable_dis_f.append(_dis_f)

        dis_comp_buff = []
        dis_comp_cands = np.array(list(itertools.product(*_usable_dis_f))).round(FLOAT_ROUND_DIGIT)
        for row in dis_comp_cands:
            if round(row.sum(), FLOAT_ROUND_DIGIT) == 1.:   # always be careful about float comparison
                dis_comp_buff.append(row)

        return np.array(dis_comp_buff)

    def contains(self, candidates_dis: Dict[str, List[float]]):
        """ Check if candidates_dis is in the space """
        return self.space.__contains__(self.space.params_to_array(candidates_dis))
    
# NOTE keep it mind the order of bo.space.keys
def bayes_opt_serial(n_init_rand = 30,
                     n_iter = 1500,
                     seed = 42):
    '''
        n_init_rand: int, number of initial random points
        n_iter: int, number of experimental iterations (outer loop) of BO, use < 200 for laptops
    '''
    id = str(uuid.uuid4())[:8]

    '''
        NOTE params is stored in sorted keys order
        If the key strings are not sorted, you need
        to change the internal implementation of
        self.register() and self.contains()
    '''
    x_name_space = ['0-C', '1-Al', '2-V', '3-Cr', '4-Mn', '5-Fe', '6-Co', '7-Ni', '8-Cu', '9-Mo']

    ''' ground truth function '''
    func = get_mo_ground_truth_func()   # multi-objective ground truth function

    ''' minimize -> maximize '''
    def to_maximize(**kwargs):
        """ x: vector of input values """
        x = np.array([kwargs[xn] for xn in x_name_space])
        return func(x)

    pbounds = dict(zip(x_name_space, COMP_LIMITS))  # NOTE

    dbo = DiscreteCompositionBO(
        f = to_maximize,
        pbounds = pbounds,
        verbose = 2,            # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state = seed,
    )

    ''' explicitly register the discrete x values, the design space infos '''
    dbo.register_compositional_limits()

    # EI utility instance
    utility = UtilityFunction(kind = "ei", xi = 0.0)

    x_order_buff = []
    bsf_buff = []

    ''' random initialization n_init_rand exps '''
    dbo.init_discrete_rand_samples(n_init_rand, seed, x_order_buff, bsf_buff)

    constraints = [
        {
            'type':'eq', 
            'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] - 1.0, 
        },
    ]

    ''' main loop of BO, after initial random exploration '''
    for i in range(n_init_rand, n_iter + 1):
        ''' inner optimization (implicit inner loop of BO) gives several continuous x '''
        continuous_candidate_s = dbo.suggest_contiuous_x(utility, constraints)
        
        ''' apply discretization and EI calculation '''
        discrete_candidate_s, ei_s = [], []
        for candidate_cont in continuous_candidate_s:
            _all_dis_combo = dbo.continuous_to_discrete(candidate_cont)
            discrete_candidate_s += _all_dis_combo.tolist()
            ei_s += utility.utility(
                _all_dis_combo, 
                gp = dbo._gp, 
                y_max = dbo.space.target.max()
            ).flatten().tolist()
        
        sorted_idx = np.argsort(ei_s)[::-1]

        ''' enumerate all surrounding discretized xs '''
        for _i in sorted_idx:
            candidate_dis = dict(zip(x_name_space, discrete_candidate_s[_i]))
            found = not dbo.contains(candidate_dis)
            if found:
                break
        assert found, 'no new candidate found'
        
        ''' update BO dbo '''
        target = to_maximize(**candidate_dis)
        dbo.register(params = candidate_dis, target = target)
        
        bsf_buff.append(dbo.max['target'])
        
        if i % 1 == 0:  # verbose print granularity
            print(id, 'iteration:', i, 'best_func_val:', round(dbo.max['target'], FLOAT_ROUND_DIGIT))
    
    print(id, 'done')
    return x_order_buff, bsf_buff

if __name__ == '__main__':
    n_init_rand = 20        # number of initial random points

    ''' parallel execution '''
    seed_list = list(range(96))
    par_res = joblib.Parallel(n_jobs = 12)(joblib.delayed(bayes_opt_serial)(n_init_rand = n_init_rand, seed = sd) for sd in seed_list)
    joblib.dump(par_res, f'bayes_opt-discrete-max_seed-{len(seed_list)}-240715-{str(uuid.uuid4())[:8]}.pkl')