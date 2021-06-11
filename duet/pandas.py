from .duet_core import *
from .duet_envs import *
import pandas as pd
import numpy as np
import copy

def read_csv(filename):
    source = DataSource(filename)
    senv = SensEnv({source: 1})
    return DuetWrapper(pd.read_csv(filename), senv, LInf())

DataFrame = pd.DataFrame

def printtype(x):
    print(f'type : {type(x)}')

k = pd.Series.to_dict

def to_dict(x):
    # printtype(x)
    if isinstance(x, DuetWrapper):
        return DuetWrapper(k(x.val), x.senv, LInf())
    else:
        return DuetWrapper(k(x), SensEnv({}), LInf())
        # return k(x)

    # y = unwrap(x)
    # r = pd.Series.to_dict(y)

    # printtype(y)
    # r = y.to_dict()
    # return DuetWrapper(r, x.senv, LInf())
    # raise ValueError('hey')

Series = copy.deepcopy(pd.Series)
Series.to_dict = to_dict
