from .duet_core import *
from functools import reduce
import math
import numpy as np

class SensEnv:
    def __init__(self, sens={}):
        self.sens = sens

    def _add_dicts(self, a, b):
        return {x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}

    def __add__(self, other):
        all_sens = self._add_dicts(self.sens, other.sens)
        return SensEnv(all_sens)

    def extend(self, var, sens):
        return self + SensEnv({var: DuetReal(sens)})

    def scale(self, amount):
        newSenses = {k : amount * v for (k, v) in self.sens.items()}
        return SensEnv(newSenses)

    def exp(self):
        newSenses = {k : np.exp(v) for (k, v) in self.sens.items()}
        return SensEnv(newSenses)

    def sqrt(self):
        newSenses = {k : np.sqrt(v) for (k, v) in self.sens.items()}
        return SensEnv(newSenses)

    def truncate(self, amount):
        newSenses = {k : amount for (k, v) in self.sens.items()}
        return SensEnv(newSenses)

    def clip(self, amount):
        newSenses = {k : amount for (k, v) in self.sens.items()}
        return SensEnv(newSenses)

    def shrug(self, epsilon, delta):
        epsilons = {k : epsilon for (k, v) in self.sens.items()}
        deltas = {k : delta for (k, v) in self.sens.items()}
        return EDPrivEnv(epsilons, deltas)

    def renyi_shrug(self, alpha, epsilon):
        alphas = {k : alpha for (k, v) in self.sens.items()}
        epsilons = {k : epsilon for (k, v) in self.sens.items()}
        return RenyiPrivEnv(alphas, epsilons)

    def get_max(self):
        vs = self.sens.values()
        if vs:
            return max(vs)
        else:
            return 0

    def __str__(self):
        senses = [f'{k}: {v}' for (k,v) in self.sens.items()]
        ss = ','.join(senses)
        return f'[{ss}]'

from abc import ABC, abstractmethod

class PrivEnv(ABC):
    @abstractmethod
    def _add_dicts(self, a, b):
        pass

    @abstractmethod
    def _max_dicts(self, a, b):
        pass

    @abstractmethod
    def penv_max(self, other):
        pass

    @abstractmethod
    def extend(self):
        pass

    @abstractmethod
    def remove(self):
        pass

class SinglePrivEnv(PrivEnv):
    def _add_dicts(self, a, b):
        pass

    def _max_dicts(self, a, b):
        pass

    def penv_max(self, other):
        pass

    def extend(self):
        pass

    def remove(self):
        pass

class DoublePrivEnv(PrivEnv):
    def __init__(self, fp={}, sp={}):
        self.fp = fp
        self.sp = sp

    def _add_dicts(self, a, b):
        return {x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}

    def _sum_costs(self):
        a = 0
        b = 0
        for i in set(self.fp):
            a += self.fp.get(i,0)
        for i in set(self.sp):
            b += self.sp.get(i,0)
        return [a,b]

    def __add__(self, other):
        all_fps = self._add_dicts(self.fp, other.fp)
        all_sps = self._add_dicts(self.sp, other.sp)
        return type(self)(all_fps, all_sps)

    def truncate(self,fp,sp):
        fps = {k : fp for (k, v) in self.fp.items()}
        sps = {k : sp for (k, v) in self.sp.items()}
        return type(self)(fps, sps)

    def _max_dicts(self, a, b):
        return {x: max(a.get(x, 0), b.get(x, 0)) for x in set(a).union(b)}

    def penv_max(self, other):
        all_fps = self._max_dicts(self.fp, other.fp)
        all_sps = self._max_dicts(self.sp, other.sp)
        return type(self)(all_fps, all_sps)

    def extend(self, var, fp , sp):
        return self + type(self)({var: DuetReal(fp)}, {var: DuetReal(sp)})

    def remove(self, var):
        e = dict(self.fp)
        d = dict(self.sp)
        try:
            e.pop(var)
            d.pop(var)
        except KeyError as ex:
            pass
        return type(self)(e, d)

    def __str__(self):
        vals = [f'{k}: ({self.fp[k]}, {self.sp[k]})' for k in self.fp.keys()]
        es = ', '.join(vals)
        return f'[{es}]'


class EpsPrivEnv(SinglePrivEnv):
    pass

class ZCPrivEnv(SinglePrivEnv):
    pass

class TCPrivEnv(DoublePrivEnv):
    pass

class RenyiPrivEnv(DoublePrivEnv):
    def ed(self,delta):
        alphas = self.fp
        epsilons = self.sp
        new_epsilon = {x: epsilons[x] + (np.log(1/delta) / (alphas[x] - 1)) for x
                       in self.sp.keys()}
        new_delta = {x: delta for x in self.sp.keys()}
        return EDPrivEnv(new_epsilon, new_delta)

    def _add_alphas(self, a, b):
        return {x: a.get(x, 0) if (a.get(x, 0) == b.get(x, 0)) else a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}

    def __add__(self, other):
        all_fps = self._add_alphas(self.fp, other.fp)
        all_sps = self._add_dicts(self.sp, other.sp)
        return type(self)(all_fps, all_sps)

    def _sum_costs(self):
        a = 0
        b = 0
        for i in set(self.fp):
            a = self.fp.get(i,0)
        for i in set(self.sp):
            b += self.sp.get(i,0)
        return [a,b]


class EDPrivEnv(DoublePrivEnv):
    def advcomp(self, k, delta_prime):
        def sc_eps(eps):
            return 2 * eps * np.sqrt(2 * k * np.log(1 / delta_prime))

        def sc_delta(delta):
            return k * delta + delta_prime

        new_epsilon = {x: sc_eps(epsilon) for (x,epsilon) in self.fp.items()}
        new_delta = {x: sc_delta(delta) for (x,delta) in self.sp.items()}
        return EDPrivEnv(new_epsilon, new_delta)


class VariantException(Exception):
    pass

class PrivacyFilterException(Exception):
    pass

class PartitionException(Exception):
    pass
