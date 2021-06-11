import math
import string
import random
from .duet_vals import *
from .duet_vals import unwrap as uw
import itertools
import string
import random
import functools
from collections.abc import Iterable



def theorem_2_11_K(epses,ε,δ):
    β = (ε * ε) / (28.04 * math.log(1/δ))
    return lemma_6_4(epses, β, δ)

# advcomp odometer
def lemma_6_4(epses, β, δ):
    t1 = sum(old_list(old_map(lambda ε: (math.exp(ε) - 1) * ε/2, epses)))
    t2a = sum(old_list(old_map(lambda ε: (ε * ε) + β, epses)))
    a = sum(old_list(old_map(lambda ε: (ε * ε), epses)))
    t2b = 1 + (math.log((a/β)+1))/2
    if δ == 0:
        δ = 0.0001
    t2c = math.log(2/δ)
    print(f't1: {t1}')
    print(f't2: {math.sqrt(2 * t2a * t2b * t2c)}')
    return t1 + math.sqrt(2 * t2a * t2b * t2c)

def penvlmax(penvl):
    x = EDPrivEnv(fp={}, sp={})
    for p in penvl:
        x = x.penv_max(p)
    return x

def iterable(obj):
    """
       :param obj:


       :return:

    """
    return isinstance(obj, Iterable)

old_list = list
old_zip = zip

def zip(x,y):
    """
       :param x:
       :param y:


       :return:

    """
    if isinstance(x,DuetWrapper) or isinstance(y,DuetWrapper):
        return DuetWrapper(old_zip(x,y),get_senv(x)+get_senv(y),get_mode(x))
    else:
        return old_zip(x,y)

def list(l):
    """
       :param l:


       :return:

    """
    l1,l2,l3 = itertools.tee(l,3)
    # TODO
    # modes = map(lambda x: x.mode.bound, old_list(l2))
    modes = old_map(lambda x: x.mode, old_list(l2))
    # modes = set(old_list(modes))
    dssources = old_map(lambda x: old_list(x.senv.sens.keys()), old_list(l3))
    fldss = old_list(itertools.chain(*old_list(dssources)))
    s = set(old_list(modes))

    if len(s) == 1 and len(set(fldss)) == 1:
        mode = s.pop()
        if hasattr(l, 'senv'):
            sns = l.senv.sens
        else:
            sns = {set(fldss).pop(): mode.bound}
        sens = sns
    else:
        mode = LInf()
        sens = {}
    r = DuetWrapper(old_list(l1),SensEnv(sens),mode)
    return r

old_dict = dict

def list2(l):
    """
       :param l:


       :return:

    """
    senv = SensEnv({'source': 1})
    return DuetWrapper(l, senv, LInf())

def dict(d):
    """
       :param d:


       :return:

    """
    senv = SensEnv({'source': 1})
    return DuetWrapper(d, senv, LInf())

L2 = L2


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
       :param size:
       :param chars:


       :return:

    """
    return ''.join(random.choice(chars) for x in range(size))

random_generator()
# random_generator(3, "6793YUIO")

old_map = map

def map(f,l):
    """
       :param f:
       :param l:


       :return:

    """
    sx  = DataSource(random_generator())
    σ   = SensEnv({sx : 1.0})
    vss = []
    xσs = []
    σs = []
    newσ = SensEnv()
    modes = []
    dssources = []
    for a in unwrap(l):
        x = f(DuetWrapper(a,σ,LInf()))
        # get the sens for "x" from each result
        s = x.senv.sens.get(sx,0)
        xσs.append(s)
        x.senv.sens.pop(sx,0)
        # get the "non-x" sens for each result
        σs.append(x.senv)
        modes.append(x.mode)
        dssources.append(x.senv.sens.keys())
        vss.append(unwrap(x))
        # parallel composition for "x"
        newσ += x.senv
    xσ  = max(xσs)
    xσp = l.senv.scale(xσ)
    # sequential composition for everything else
    s  = newσ + xσp
    f = DuetWrapper(vss, s ,LInf())
    fldss = old_list(itertools.chain(*old_list(dssources)))
    s = set(old_list(modes))
    if len(s) == 1 and len(set(fldss)) == 1:
        mode = s.pop()
        if hasattr(f, 'senv'):
            sns = f.senv.sens
        else:
            sns = {set(fldss).pop(): mode.bound}
        sens = sns
    else:
        mode = LInf()
        sens = {}
    # return r # return the result list with the new σ
    r = DuetWrapper(vss,SensEnv(sens),mode)
    return r


def unwrap(x):
    """
       :param x:


       :return:

    """
    return uw(x)

def mode_switch(m1,m2):
    """
       :param m1:
       :param m2:


       :return:

    """
    def mode_switch(func):
        """
           :param func:


           :return:

        """
        def wrapper(v,b,*args,**kwargs):
            """
               :param val:
               :param **kwargs:


               :return:

            """
            d = func(v,b,*args,**kwargs)
            if d.mode != m1() :
                raise ValueError(f'mode_switch: unexpected mode {d.mode} instead of {m1}')
            else:
                d.senv = d.senv.clip(b)
                d.mode = m2(b)
            return d
        return wrapper
    return mode_switch

@mode_switch(LInf,L2)
def L2_clip(v, b):
    """
       :param v:
       :param b:


       :return:

    """
    norm = np.linalg.norm(v, ord=2)
    if norm.val > b:
        return b * (v / norm)
    else:
        return v

def debug(msg):
    """
       :param msg:
       :param delta:


       :return:

    """
    # print(f'DEBUG: {msg}')
    pass
