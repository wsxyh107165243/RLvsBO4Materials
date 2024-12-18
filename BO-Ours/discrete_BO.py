'''
    Discrete Bayesian Optimization.

    This script is a demonstration of the Discrete Bayesian 
    Optimization (DisBO) algorithm on several test function
    with varying dimensionalities.

    The DisBO algorithm is a modification of the traditional
    Bayesian Optimization (BO) algorithm that is designed
    to handle continuous input spaces. The algorithm works
    by discretizing the design space and then performing
    BO on the discretized space.

    Several implementation details have been modified to
    improve the performance of the algorithm. These include:
        1.  Discretization around the found *x_continuous
        2.  Enlarged inner loop of BO (argmax\-(x)f_acq) 
            for better exploration of *x_discrete

    @author: <xianyuehui@stu.xjtu.edu.cn>
'''

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

from utils import *

FLOAT_ROUND_DIGIT = 4

''' Just ignore numerous sklearn warnings '''
def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

class DiscreteBayesianOptimization(BayesianOptimization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def register_discrete_x(self, x_discrete, x_min, x_max, N):
        """ Register the discrete x """
        if isinstance(x_discrete, dict):
            self._x_discrete = x_discrete
            self._x_min = x_min
            self._x_max = x_max
            self._N = N
            self._min_interval = round((x_max - x_min) / (N - 1), FLOAT_ROUND_DIGIT)
        else:
            raise ValueError("x_discrete must be a dictionary of dicrete x values for each dimension")
        
    def init_rand_N(self, n_init_rand: int, bsf_buff: List):
        """ Initialize the random N points """
        assert self._x_discrete is not None, 'Discrete x space is not initialized.'

        while len(self.space) < n_init_rand:
            candidate_dis = {xk: np.random.choice(self._x_discrete[xk]) for xk in self.space.keys}
            if self.contains(candidate_dis):
                continue
            target = self.space.target_func(**candidate_dis)
            self.register(params = candidate_dis, target = target)
        
            bsf_buff.append(self.max['target'])

    @ignore_warnings
    def suggest_contiuous_x(self, utility_function) -> List:
        """ Most promising point to probe next """
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        ac = utility_function.utility
        gp = self._gp
        y_max = self._space.target.max()
        bounds = self._space.bounds
        random_state = self._random_state
        # default hyperparameters for acq_max
        n_warmup = 10000    # number of times to randomly sample the acquisition function
        '''
            Number of times to run scipy.minimize. The default value in bayes_opt is 10.
            However, we believe that using a number that scales up with dimension would 
            be a better choice. This requires more computational cost but is necessary,
            especially for high-dimensional problems (or test functions).
        '''
        n_iter = 16 * self.space.dim

        ''' Main body of continuous_x inner loop (optimization) of BO '''
        continuous_x_buff = []
        # Warm up with random points
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_warmup, bounds.shape[0]))
        ys = ac(x_tries, gp=gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        continuous_x_buff.append(x_max)

        # Explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_iter, bounds.shape[0]))

        to_minimize = lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max)

        for x_try in x_seeds:
            '''
                Find the minimum of minus the acquisition function.
                L-BFGS-B is a quasi-Newton method that uses a limited-memory BFGS method.
                It is able to use numerical gradient for handling continuous x.
            '''
            res = minimize(lambda x: to_minimize(x),
                        x_try,
                        bounds=bounds,
                        method="L-BFGS-B")

            # See if success
            if not res.success:
                continue

            tmp_continuous_x = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            continuous_x_buff.append(tmp_continuous_x)

        return continuous_x_buff

    def round_neighbors(self, x):
        """ discretization for x """
        x_low_neighbor = math.floor(x / self._min_interval) * self._min_interval
        x_high_neighbor = math.ceil(x / self._min_interval) * self._min_interval
        return np.unique(np.clip([x_low_neighbor, x_high_neighbor], self._x_min, self._x_max))

    def continuous_to_discrete(self, candidate_continuous: List[float]) -> np.ndarray:
        """ Convert the continuous candidate to discrete candidates """
        _usable_dis_f = [self.round_neighbors(x_cont) for x_cont in candidate_continuous]
        return np.array(list(itertools.product(*_usable_dis_f)))

    def contains(self, candidates_dis: Dict[str, List[float]]):
        """ Check if candidates_dis is in the space """
        return self.space.__contains__(self.space.params_to_array(candidates_dis))

def bayes_opt_serial(func_name: str = 'ackley', 
                     D: int = 4,
                     n_init_rand: int = 112,
                     n_iter = 300):
    '''
        func_name: str, name of the function to optimize
        D: int, dimension of the function to optimize
        n_init_rand: int, number of initial random points
        n_iter: int, number of experimental iterations (outer loop) of BO, use < 200 for laptops
    '''
    id = str(uuid.uuid4())[:8]

    func = func_params[func_name]['func']
    x_min, x_max = func_params[func_name]['x_min'], func_params[func_name]['x_max']
    N = func_params[func_name]['act_dim']
    x_name_space = ['x' + str(i) for i in range(D)]

    ''' minimize -> maximize '''
    def to_maximize(**kwargs):
        """ x: vector of input values """
        x = np.array([kwargs[xn] for xn in x_name_space])
        return - func(x)

    pbounds = dict(zip(x_name_space, [(x_min, x_max) for _ in x_name_space]))

    dbo = DiscreteBayesianOptimization(
        f = to_maximize,
        pbounds = pbounds,
        verbose = 2,    # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state = random.randint(0, 999),
    )

    ''' register the discrete x values, the design space infos '''
    dbo.register_discrete_x(
        x_discrete = {xn: np.linspace(x_min, x_max, N) for xn in x_name_space},
        x_min = x_min,
        x_max = x_max,
        N = N
    )

    # EI utility instance
    utility = UtilityFunction(kind = "ei", xi = 0.0)

    bsf_buff = []

    ''' random initialization n_init_rand exps '''
    dbo.init_rand_N(n_init_rand, bsf_buff)

    ''' main loop of BO, after initial random exploration '''
    for i in range(n_init_rand, n_iter + 1):
        ''' inner optimization (implicit inner loop of BO) gives several continuous x '''
        continuous_candidate_s = dbo.suggest_contiuous_x(utility)
        
        ''' apply discretization and EI calculation '''
        ''' TODO: use set() to select unique candidates '''
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
            
        if abs(dbo.max['target'] - 0.0) < 1e-4:
            break
    
    print(id, func_name, func_dim, 'done')
    return bsf_buff

# bayes_opt_serial('rastrigin', 5), exit()

if __name__ == '__main__':
    func_name = 'rastrigin'         # name for test functions, ackley, rastrigin
    par_N = 48                      # number of parallel experiment

    for func_dim in range(3, 10):   # for ackley, func dimension of 4 is recommended in SMOKE_TEST
        par_res = joblib.Parallel(n_jobs = -1)(joblib.delayed(bayes_opt_serial)(func_name = func_name, D = func_dim) for _ in range(par_N))
        joblib.dump(par_res, f'bayes_opt-{func_name}-{func_dim}-discrete-{par_N}-{str(uuid.uuid4())[:8]}.pkl')
        print('Done:', func_name, func_dim)