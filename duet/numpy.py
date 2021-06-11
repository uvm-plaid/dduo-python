from .duet_core import *
from .duet_envs import *
import pandas as pd
import numpy as np

def load(filename):
    source = DataSource(filename)
    senv = SensEnv({source: 1})
    return DuetWrapper(np.load(filename), senv, LInf())

def zeros(x):
    if isinstance(x, DuetWrapper):
        y = unwrap(x)
        r = np.zeros(y)
        return DuetWrapper(r, x.senv, LInf())
    else:
        return np.zeros(x)

# def zeros(x):
#     y = unwrap(x)
#     r = np.zeros(y)
#     return DuetWrapper(r, x.senv.truncate(0), LInf())

def exp(x):
    if isinstance(x,DuetWrapper):
        y = unwrap(x)
        r = np.exp(y)
        if hasattr(x, 'senv'):
            return DuetWrapper(r, x.senv, LInf())
        else:
            return DuetWrapper(r, SensEnv({}), LInf())
    else:
        return np.exp(x)

# def exp(x):
#     if isinstance(x,DuetWrapper):
#         y = unwrap(x)
#         r = np.exp(y)
#         if hasattr(x, 'senv'):
#             return DuetWrapper(r, x.senv.exp(), LInf())
#         else:
#             return DuetWrapper(r, SensEnv({}), LInf())
#     else:
#         return np.exp(x)

def abs(x):
    y = unwrap(x)
    r = np.abs(y)
    return DuetWrapper(r, get_senv(x), LInf())

def sign(x):
    y = unwrap(x)
    r = np.sign(y)
    return DuetWrapper(r, get_senv(x), LInf())

# def sign(x):
#     y = unwrap(x)
#     r = np.sign(y)
#     return DuetWrapper(r, get_senv(x).truncate(2), LInf())

def sqrt(x):
    y = unwrap(x)
    r = np.sqrt(x)
    return DuetWrapper(r, get_senv(x), LInf())

# def sqrt(x):
#     y = unwrap(x)
#     r = np.sqrt(x)
#     return DuetWrapper(r, get_senv(x).sqrt(), LInf())

def dot(x,y):
    a = unwrap(x)
    b = unwrap(y)
    return DuetWrapper(np.dot(a,b), get_senv(x)+get_senv(y), LInf())

# def dot2(x,y):
#     a = unwrap(x)
#     b = unwrap(y)
#     return DuetWrapper(np.dot(a,b), length(x)*(get_senv(x)*get_senv(y)), LInf())

def subtract(x,y):
    if isinstance(x, DuetWrapper) or isinstance(y, DuetWrapper):
        a = unwrap(x)
        b = unwrap(y)
        return DuetWrapper(np.subtract(a,b), get_senv(x)+get_senv(y), LInf())
    else:
        return np.subtract(x, y)

def sum(x,*args,**kwargs):
    if isinstance(x,DuetWrapper):
        if isinstance(x.mode, L2):
            y = unwrap(x)
            r = np.sum(y,*args,**kwargs)
            return DuetWrapper(r, x.senv.truncate(x.mode.bound), x.mode)
        else:
            y = unwrap(x)
            r = np.sum(y,*args,**kwargs)
            return DuetWrapper(r, SensEnv(), x.mode)
    else:
        return np.sum(x,*args,**kwargs)

def sum2(x,*args,**kwargs):
    if isinstance(x,DuetWrapper):
        if isinstance(x.mode, L2):
            y = unwrap(x)
            r = np.sum(y,*args,**kwargs)
            return DuetWrapper(r, x.senv.truncate(x.mode.bound), x.mode)
        else:
            y = unwrap(x)
            r = np.sum(y,*args,**kwargs)
            return DuetWrapper(r, SensEnv(), x.mode)
    else:
        return np.sum(x,*args,**kwargs)


# sum = np.sum

linalg = np.linalg
random = np.random
old_norm = np.linalg.norm
where = np.where
# abs = np.abs

def where(a,b,c):
    return DuetWrapper(np.where(a.val,b.val,c.val),a.senv+b.senv+c.senv,LInf())

def where2(a):
    return DuetWrapper(np.where(a.val),a.senv,LInf())

def where3(a):
    return np.where(a)

def norm(x, ord=None, axis=None):
    xp = unwrap(x)
    new_senv = get_senv(x).scale(float('inf'))
    return DuetWrapper(old_norm(xp, ord, axis), new_senv, LInf())

linalg.norm = norm

def array_split(arr, n):
    arrp = unwrap(arr)
    split_arr = np.array_split(arrp, n)
    sv = get_senv(arr)
    wrapped_arr = np.array([DuetWrapper(x, sv, LInf()) for x in split_arr])
    return wrapped_arr
