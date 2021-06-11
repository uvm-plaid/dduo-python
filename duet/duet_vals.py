import math
from functools import partial
from .duet_core import *
from .duet_envs import *
import pandas as pd
import numpy as np
from _collections_abc import dict_items

class DistanceMetric():
    def __init__(self, bound=float('inf')):
        self.bound = bound

    def __hash__(self):
        return hash(self.bound)

    def __eq__(self, other):
        return other.bound == self.bound

class L1(DistanceMetric):
    pass

class L2(DistanceMetric):
    pass

class LInf(DistanceMetric):
    pass

# Data source
class DataSource:
    def __init__(self, desc):
        self.desc = desc

    def __str__(self):
        return f'DataSource({self.desc})'

    def __eq__(self, other):
        return isinstance(other, DataSource) and other.desc == self.desc

    def __hash__(self):
        return hash(self.desc)

# Utilities for the Duet Wrapper
def unwrap(obj):
    if isinstance(obj, DuetWrapper):
        return unwrap(obj.val)
    else:
        return obj

def get_senv(obj):
    if isinstance(obj, DuetWrapper):
        return obj.senv
    else:
        return SensEnv()

def get_mode(obj):
    if isinstance(obj, DuetWrapper):
        return obj.mode
    else:
        return LInf()


# These are the sensitivity "transformers" used in the lookup table below
def sens_id(senv, mode, args):
    return senv, mode

def sens_inf(senv, mode, args):
    #print(f'called inf {senv}')
    return senv.scale(float('inf')), mode


def sens_clip(senv, mode, args):
    lower, upper = args
    assert mode == LInf()
    return senv.scale(upper - lower), L1()

def sens_sum(senv, mode, args):
    assert mode == L1()
    return senv, mode

def sens_getitem(senv, mode, args):
    # TODO: this should probably use the args in some cases
    return senv, mode

# This is the "lookup table" for sensitivities of methods
sens_effects = {
    np.ndarray:{
        'size': sens_id,
        'copy': sens_id,
        'shape': sens_id,
        'sum': sens_id,
        'astype': sens_id,
        'flatten': sens_id,
        'dot': sens_id,
        '__getitem__': sens_getitem
    },
    pd.Series: {
        'shape': sens_id,
        'value_counts': sens_id,
        'to_dict': sens_id,
        'clip':  sens_clip,
        'sum':   sens_sum
    },
    pd.DataFrame: {
        'shape': sens_id, # wrong, cols should have sens 0
        'count': sens_id,
        '__getitem__': sens_getitem,
        'values': sens_id
    },
    tuple: {
        '__getitem__': sens_getitem
    },
    dict: {
        'update': sens_id,
        'keys': sens_id,
        '__contains__': sens_id,
        'items': sens_id
    },
    dict_items: {
        '__getitem__': sens_getitem,
    }
}


class DuetWrapper:
    global sens_effects


    def __init__(self, obj, senv, mode=LInf()):
        if isinstance(obj, DuetWrapper):
            self.senv = senv + obj.senv
            self.val = obj.val
            self.mode = mode
        else:
            self.senv = senv
            self.val = obj
            self.mode = mode


    def __getattr__(self, name):
        typ = type(self.val)
        attr = getattr(self.val, name)

        try:
            sens_effects['duet.duet_vals.@DuetWrapper'] = {'items': sens_id}

            sens_mod = sens_effects[typ][name]
        except:
            sens_mod = sens_id
            # raise Exception(f'No sensitivity definition for {typ}.{name}')

        if callable(attr):
            def _missing(*args, **kwargs):
                new_senv = self.senv
                for a in args:
                    if isinstance(a, DuetWrapper):
                        new_senv = new_senv + a.senv

                result = attr(*args, **kwargs)

                new_senv, new_mode = sens_mod(new_senv, self.mode, args)

                return DuetWrapper(result, new_senv, new_mode)
            return _missing
        else:
            new_senv, new_mode = sens_mod(self.senv, self.mode, [])

            return DuetWrapper(attr, new_senv, new_mode)

    def __getitem__(self, key):
        name = '__getitem__'
        typ = type(self.val)

        try:
            sens_mod = sens_effects[typ][name]
        except:
            sens_mod = sens_id
            # raise Exception(f'No sensitivity definition for {typ}.{name}')

        new_senv, new_mode = sens_mod(self.senv, self.mode, [key])
        if hasattr(self.val, '__getitem__'):
            result = self.val.__getitem__(unwrap(key))
        else:
            result = list(self.val).__getitem__(unwrap(key))

        return DuetWrapper(result, new_senv, new_mode)

    def __bool__(self):
        raise RuntimeError('Sensitive values cannot be used in guard position of a conditional')

    def __str__(self):
        sub = str(type(self.val))
        return f'DuetWrapper({sub}, {self.senv}, {self.mode})'

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        new_env = self.senv.scale(float('inf'))
        return DuetWrapper(self.val == unwrap(other), new_env, self.mode)

    def __abs__(self):
        return DuetWrapper(abs(self.val), self.senv, self.mode)

    def __gt__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val > unwrap(other), new_env, newMode)

    def __lt__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val < unwrap(other), new_env, newMode)

    def __le__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val <= unwrap(other), new_env, newMode)

    def __ge__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val >= unwrap(other), new_env, newMode)

    def __add__(self, other):
        return DuetWrapper(self.val + unwrap(other), self.senv + get_senv(other), self.mode)

    def __sub__(self, other):
        return DuetWrapper(self.val - unwrap(other), self.senv + get_senv(other), self.mode)


    def __rsub__(self, other):
        return DuetWrapper(unwrap(other)-self.val, self.senv + get_senv(other), self.mode)

    def __radd__(self, other):
        return DuetWrapper(self.val + unwrap(other), self.senv + get_senv(other), self.mode)

    def __and__(self, other):
        return DuetWrapper(self.val & unwrap(other), self.senv + get_senv(other), self.mode)


    # TODO
    def __rmul__(self, other):
        return DuetWrapper(self.val * unwrap(other), self.senv + get_senv(other), self.mode)

    def __rtruediv__(self, other):
        return DuetWrapper(unwrap(other)/self.val, self.senv + get_senv(other), self.mode)

    def __itruediv__(self, other):
        return DuetWrapper(unwrap(other)/self.val, self.senv + get_senv(other), self.mode)

    def __rfloordiv__(self, other):
        return DuetWrapper(unwrap(other)//self.val, self.senv + get_senv(other), self.mode)

    def __ifloordiv__(self, other):
        return DuetWrapper(unwrap(other)//self.val, self.senv + get_senv(other), self.mode)

    def __int__(self):
        return int(self.val)

    # def to_dict(self):
    #     y = unwrap(self)
    #     r = pd.Series.to_dict(y)
    #     return DuetWrapper(r, self.senv, LInf())

    def __neg__(self):
        return DuetWrapper(-1 * (self.val), self.senv, self.mode)

    def __float__(self):
        return float(self.val)

    def __truediv__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = self.senv.scale(other)
        else:
            new_env = self.senv.scale(float('inf'))
        # print(self.val)
        # print(unwrap(other))
        return DuetWrapper(self.val / unwrap(other), new_env, self.mode)

    def __floordiv__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = self.senv.scale(other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val // unwrap(other), new_env, self.mode)


    def __matmul__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = self.senv.scale(other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val @ unwrap(other), new_env, self.mode)

    def __mul__(self, other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = self.senv.scale(other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val * unwrap(other), new_env, self.mode)

    def __rmul__(self, other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = self.senv.scale(other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val * unwrap(other), new_env, self.mode)
