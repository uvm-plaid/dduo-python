import numpy as np
import math
from .duet_vals import *
from .duet_vals import unwrap as uw
#from .pandas import *
from contextlib import contextmanager
import copy
from .utilities import *
import itertools
import string
import random
import functools
from collections.abc import Iterable

class DDAccountant():
    def __init__(self, ods = [], fs = [], advods = [], advfs = [] ):
        """
        Initializes the DDuo Accountant object

        Parameters:
            ods (list): list of regular odometers
            fs (list): list of regular filters
            advods (list): list of advanced odometers
            advfs (list): list of advanced filters

        Returns:
            DDAccountant: initialized object
        """
        self.ods = ods
        self.fs = fs
        self.advods = advods
        self.advfs = advfs

    def register_odometer(self,od):
        """
        Function which registers a regular odometer

        Parameters:
            od (Object): regular odometer
        """
        self.ods.append(od)

    def unregister_odometer(self,od):
        """
        Function which unregisters a regular odometer

        Parameters:
            od (Object): regular odometer
        """
        self.ods.remove(od)

    def register_filter(self,ftr):
        """
        Function which registers a regular filter

        Parameters:
            ftr (Object): regular odometer
        """
        self.fs.append(ftr)

    def unregister_filter(self,ftr):
        """
        Function which unregisters a regular filter

        Parameters:
            ftr (Object): regular odometer
        """
        self.fs.remove(ftr)

    def register_advod(self,advod):
        """
        Function which registers an advanced odometer

        Parameters:
            advod (Object): advanced odometer
        """
        self.advods.append(advod)

    def unregister_advod(self,advod):
        """
        Function which unregisters an advanced odometer

        Parameters:
            advod (Object): advanced odometer
        """
        self.advods.remove(advod)

    def register_advf(self,advftr):
        """
        Function which registers an advanced filter

        Parameters:
            advftr (Object): advanced filter
        """
        self.advfs.append(advftr)

    def unregister_advf(self,advftr):
        """
        Function which unregisters an advanced filter

        Parameters:
            advftr (Object): advanced filter
        """
        self.advfs.remove(advftr)

    def __str__(self):
        return f'odometers:{self.ods},filters:{self.fs}'

class EDOdometer():
    def __enter__(self):
        global accountant
        accountant.register_odometer(self)
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        accountant.unregister_odometer(self)

    def __init__(self):
        """
        Function which initializes an EDOdometer
        """
        self.privacy_env = EDPrivEnv(fp={}, sp={})

    def add(self, penv):
        """
        Function which increments privacy cost

        Parameters:
            penv (Object): ED privacy environment
        """
        self.privacy_env = self.privacy_env + penv

    def __str__(self):
        return f'({self.privacy_env})'

class RenyiDP():
    def __enter__(self):
        global accountant
        accountant.register_odometer(self)
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        accountant.unregister_odometer(self)

    def __init__(self, delta):
        """
        Function which initializes Renyi conversion

        Parameters:
            delta (float): privacy parameter
        """
        self.privacy_env = RenyiPrivEnv(fp={}, sp={})
        self.delta = delta

    def add(self, penv):
        self.privacy_env = self.privacy_env + penv

    def __str__(self):
        return f'({self.privacy_env.ed(self.delta)})'

class AdvEDFilterObj():
    def __enter__(self):
        global accountant
        accountant.register_advf(self)
        self.env = EDPrivEnv(fp={}, sp={})
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        if self.d0 > self.delta/2 or theorem_2_11_K([self.e0],self.eps,self.delta) > self.eps:
            raise PrivacyFilterException()
        accountant.unregister_advf(self)

    def __init__(self, eps, delta):
        """
        Function which initializes AdvEDFilterObj

        Parameters:
            eps (float): privacy parameter
            delta (float): privacy parameter
        """
        self.eps = eps
        self.delta = delta
        self.e0 = 0
        self.d0 = 0
        self.env = None

    def pay(self, eps, delta):
        self.e0 = self.e0 + eps
        self.d0 = self.d0 + delta
        return self

    def split(self):
        """
        Function which splits privacy budget

        Returns:
            list: list of partitioned filters
        """
        return [EDFilterObj(self.eps/2,self.delta/2),EDFilterObj(self.eps/2,self.delta/2)]

    def partition(self,ratios):
        """
        Function which partitions privacy budget

        Parameters:
            ratios (list): ratios to partition by

        Returns:
            list: list of partitioned filters
        """
        if sum(ratios) != 1.0:
            raise PartitionException
        res = []
        for r in ratios:
            res.append(EDFilterObj(self.eps/r,self.delta/r))
        return res

class RenyiOdometer():
    def __enter__(self):
        global accountant
        accountant.register_odometer(self)
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        accountant.unregister_odometer(self)

    def __init__(self, Delta, privacy_env = RenyiPrivEnv(fp={}, sp={})):
        """
        Function which initializes Renyi odometer

        Parameters:
            Delta (tuple): privacy parameters (alpha,epsilon)
        """
        self.privacy_env = privacy_env
        alpha, epsilon = Delta
        self.total_cost = epsilon
        self.alpha = alpha
        self.epsilon = epsilon
        self.filter = RenyiFilterObj(alpha, epsilon)

    def new(self,name,fp,sp):
        return type(self)(RenyiPrivEnv({name:fp},{name:sp}))

    def add(self, penv):
        """
        Function which increments privacy cost

        Parameters:
            penv (Object): Renyi privacy environment
        """
        alpha, eps = penv._sum_costs()
        try:
            self.filter.pay(self.alpha, eps)
            self.total_cost += eps
        except PrivacyFilterException:
            self.filter = RenyiFilterObj(self.alpha, self.epsilon)
            self.filter.pay(alpha, eps)
            self.total_cost += eps


    def __str__(self):
        return f'({self.alpha}, {self.total_cost})'

# this is only really used for implementing the odometer
# should maybe be a "private class"
class RenyiFilterObj():
    def __enter__(self):
        global accountant
        accountant.register_filter(self)
        self.env = RenyiPrivEnv(fp={}, sp={})
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        env1 = self.env
        tuple = env1._sum_costs()
        if tuple[0] > self.alpha or tuple[1] > self.eps:
            raise PrivacyFilterException()
        accountant.unregister_filter(self)

    def __init__(self, alpha, eps):
        """
        Function which initializes Renyi filter

        Parameters:
            alpha: privacy parameter
            eps: privacy parameter
        """
        self.alpha = alpha
        self.eps = eps
        self.env = None

    def pay(self, alpha, eps):
        """
        Function which increments Renyi filter cost

        Parameters:
            alpha: privacy parameter
            eps: privacy parameter
        """

        if self.alpha != alpha:
            raise PrivacyFilterException()
        self.eps = self.eps - eps
        if self.alpha < 0 or self.eps < 0:
            raise PrivacyFilterException()
        return self

    def split(self):
        return [RenyiFilterObj(self.alpha,self.eps/2),RenyiFilterObj(self.alpha,self.eps/2)]

    def partition(self,ratios):
        if sum(ratios) != 1.0:
            raise PartitionException
        res = []
        for r in ratios:
            res.append(RenyiFilterObj(self.alpha,self.eps/r))
        return res

class AdvEdOdometer():
    def __enter__(self):
        global accountant
        accountant.register_advod(self)
        return self

    def __exit__(self, typ, val, traceback):
        global accountant
        accountant.unregister_advod(self)

    def __init__(self):
        self.privacy_env = EDPrivEnv(fp={}, sp={})

    def add(self, penv):
        self.privacy_env = self.privacy_env + penv

    def __str__(self):
        epses = self.privacy_env.fp
        deltas = self.privacy_env.sp
        new_epsilon = {x: lemma_6_4([epses[x]], 0.0001, sum([deltas[x]])/2) for x
                       in self.privacy_env.sp.keys()}
        new_delta = {x: sum([deltas[x]])/2 for x in self.privacy_env.sp.keys()}
        return EDPrivEnv(new_epsilon, new_delta).__str__()

accountant = DDAccountant()

def add_privacy_cost(epsilon, delta, senv):
    """
    Function which allows 3rd party libraries to instrument privacy costs

    Parameters:
        eps (float): privacy parameter
        delta (float): privacy parameter
        senv (dict): sensitivity environment
    """
    global accountant

    for f in accountant.fs:
        f.pay(epsilon,delta)

    for f in accountant.advfs:
        f.pay(epsilon,delta)

    for o in accountant.ods:
        o.add(senv.renyi_shrug(epsilon,delta))

    for o in accountant.advods:
        o.add(senv.shrug(epsilon,delta))

def print_privacy_cost():
    for o in accountant.ods:
        print(o)

    for o in accountant.advods:
        print(o)


@contextmanager
def EDFilter(eps,delta):
    """
    Function which performs the dynamic filter for the ED Mechanism

    Parameters:
        eps (float): privacy parameter
        delta (float): privacy parameter

    """
    try:
        env = EDPrivEnv(fp={}, sp={})
        for o in accountant.ods:
            if isinstance(o, EDOdometer):
                env = env + o.privacy_env
        yield None
    finally:
        tuple = env._sum_costs()
        if tuple[0] > eps or tuple[1] > delta:
            raise PrivacyFilterException()

@contextmanager
def RenyiFilter(alpha,eps):
    """
    Function which performs the dynamic filter for the Renyi Gaussian Mechanism

    Parameters:
        alpha (float): privacy parameter
        eps (float): privacy parameter

    """
    try:
        env = RenyiPrivEnv(fp={}, sp={})
        for o in accountant.ods:
            if isinstance(o, RenyiOdometer):
                env = env + o.privacy_env
        yield None
    finally:
        tuple = env._sum_costs()
        if tuple[0] > alpha or tuple[1] > eps:
            raise PrivacyFilterException()

def gauss(val,**kwargs):
    """
    Function which performs the Gaussian Mechanism

    Parameters:
        val (float): numeric value to privatize
        kwargs (dict): privacy parameters
        kwargs['ε']: privacy parameter
        kwargs['δ']: privacy parameter

    Returns:
        float: val with added Guassian noise
    """
    if 'ε' in kwargs:
        eps = kwargs['ε']
    elif 'epsilon' in kwargs:
        eps = kwargs['epsilon']
    if 'δ' in kwargs:
        dlta = kwargs['δ']
    elif 'delta' in kwargs:
        dlta = kwargs['delta']
    t = type(val)
    if t == DuetWrapper: # DPVal:
        debug('Running the Gaussian mechanism')
        debug(val.senv)

        sensitivity = val.senv.get_max()
        debug(f'Maximum sensitivity: {sensitivity}')

        sigma = np.sqrt(2*np.log(1.25/dlta)) * sensitivity / eps
        noise = np.random.normal(loc=0, scale=sigma)

        for f in accountant.fs:
            f.pay(eps,dlta)

        for f in accountant.advfs:
            f.pay(eps,dlta)

        for o in accountant.ods:
            o.add(val.senv.shrug(eps,dlta))

        for o in accountant.advods:
            o.add(val.senv.shrug(eps,dlta))

        return val.val + noise
    else:
        raise ValueError(f'Gauss: Unable to analyze value {val} of type {t}')

def renyi_gauss(val,**kwargs):
    """
    Function which performs the Renyi Gaussian Mechanism

    Parameters:
        val (float): numeric value to privatize
        kwargs (dict): privacy parameters
        kwargs['α']: privacy parameter
        kwargs['ε']: privacy parameter

    Returns:
        float: val with added Guassian noise
    """

    if 'ε' in kwargs:
        eps = kwargs['ε']
    elif 'epsilon' in kwargs:
        eps = kwargs['epsilon']
    if 'α' in kwargs:
        al = kwargs['α']
    elif 'alpha' in kwargs:
        al = kwargs['alpha']
    t = type(val)
    if t == DuetWrapper: # DPVal:
        debug('Running the Renyi Gaussian mechanism')
        debug(val.senv)

        sensitivity = val.senv.get_max()
        debug(f'Maximum sensitivity: {sensitivity}')

        sigma = np.sqrt((sensitivity**2 * al) / (2 * eps))
        noise = np.random.normal(loc=0, scale=sigma)

        for f in accountant.fs:
            f.pay(al,eps)

        for o in accountant.ods:
            o.add(val.senv.renyi_shrug(al, eps))

        return val.val + noise
    else:
        return val

def renyi_gauss_vec(val,**kwargs):
    """
    Function which performs the Guassian Mechanism on vectors

    Parameters:
        val (list): list of numeric values to privatize
        kwargs (dict): privacy parameters
        kwargs['α']: privacy parameter
        kwargs['ε']: privacy parameter

    Returns:
        list: val with added Laplacian noise
    """

    if 'ε' in kwargs:
        eps = kwargs['ε']
    elif 'epsilon' in kwargs:
        eps = kwargs['epsilon']
    if 'α' in kwargs:
        al = kwargs['α']
    elif 'alpha' in kwargs:
        al = kwargs['alpha']
    t = type(val)
    if t == DuetWrapper:
        debug('Running the Renyi Gaussian mechanism')
        debug(val.senv)

        sensitivity = val.senv.get_max()
        debug(f'Maximum sensitivity: {sensitivity}')

        valp = unwrap(val)
        sigma = np.sqrt((sensitivity**2 * al) / (2 * eps))
        noise = np.random.normal(loc=0, scale=sigma, size=len(valp))

        for f in accountant.fs:
            f.pay(al,eps)

        for o in accountant.ods:
            o.add(val.senv.renyi_shrug(al, eps))

        return valp + noise
    else:
        raise ValueError(f'RenyiGauss: Unable to analyze value {val} of type {t}')


def laplace(val, **kwargs):
    """
    Function which performs the Laplacian Mechanism

    Parameters:
        val (float): numeric value to privatize
        kwargs (dict): privacy parameters
        kwargs['ε']: privacy parameter

    Returns:
        float: val with added Laplacian noise
    """

    if 'ε' in kwargs:
        eps = kwargs['ε']
    elif 'epsilon' in kwargs:
        eps = kwargs['epsilon']
    debug('Running the Laplace mechanism')
    debug(val.senv)

    sensitivity = val.senv.get_max()
    debug(f'Maximum sensitivity: {sensitivity}')

    scale = sensitivity / eps
    noise = np.random.laplace(loc=0, scale=scale)

    for f in accountant.fs:
        f.pay(eps,0)

    for f in accountant.advfs:
        f.pay(eps,0)

    for o in accountant.ods:
        o.add(val.senv.shrug(eps,0))

    for o in accountant.advods:
        o.add(val.senv.shrug(eps,0))


    # check if value is a list for parallel composition
    if hasattr(val.val, '__len__'):
        size = len(val.val)
        return val.val + np.random.laplace(loc=0, scale=scale, size=size)

    return val.val + noise

def report_noisy_max(val, R, u, epsilon):
    """
    Function which performs the exponential mechanism/report noisy max

    Parameters:
        val (string): object attribute
        R (list): dataset
        u (function): scoring function
        epsilon (float): privacy parameter

    Returns:
        Object: privately selected object
    """

    t = type(val)
    if str(t) == "<class 'duet.pandas.DPSeries'>" or t == DuetWrapper:
        # Calculate the score for each element of R
        scores = [u(val, r) for r in R]
        raw_scores = [s.val for s in scores]

        # Get max sensitivity
        sensitivities = [s.senv.get_max() for s in scores]
        sensitivity = max(sensitivities)

        # Add noise to each score
        debug(f'Maximum sensitivity: {sensitivity}')

        scale = sensitivity / epsilon
        noisy_scores = [score + np.random.laplace(loc=0, scale=scale) for score in raw_scores]

        # Find the index of the maximum score
        max_idx = np.argmax(noisy_scores)

        for f in accountant.fs:
            f.pay(epsilon,0)

        for f in accountant.advfs:
            f.pay(epsilon,0)

        for o in accountant.ods:
            o.add(scores[0].senv.shrug(epsilon,0))

        for o in accountant.advods:
            o.add(scores[0].senv.shrug(epsilon,0))

        # Return the element corresponding to that index
        return R[max_idx]

    else:
        raise ValueError(f'Unable to analyze value {val} of type {t}')

# preserves epsilon-differential privacy
def above_threshold(queries, df, T, epsilon):
    """
    Function which performs the sparse vector technique/above threshold

    Parameters:
        queries (list): list of queries
        df (dict): dataset
        T (float): threshold
        epsilon (float): privacy parameter

    Returns:
        int: index of the selected query
    """

    T_hat = T + np.random.laplace(loc=0, scale = 2/epsilon)
    query_result = None

    for idx, q in enumerate(queries):
        query_result = q(df)

        nu_i = np.random.laplace(loc=0, scale = 4/epsilon)

        if query_result.val + nu_i >= T_hat:
            for f in accountant.fs:
                f.pay(epsilon,0)

            for f in accountant.advfs:
                f.pay(epsilon,0)

            for o in accountant.ods:
                o.add(query_result.senv.shrug(epsilon,0))

            for o in accountant.advods:
                o.add(query_result.senv.shrug(epsilon,0))
            # Return the idx
            return idx

    # Return the index of the last element
    return -1
