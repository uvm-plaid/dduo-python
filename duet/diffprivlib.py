from .duet_core import *
from .duet_envs import *

import diffprivlib.models as models
from sklearn.metrics import accuracy_score

import random
import string

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def genid(s):
    return abs(hash(s)) % (10 ** 8)

def score(a,b):
    senv = SensEnv({'source': 1})
    return DuetWrapper(accuracy_score(a.val,b), senv, LInf())
    # return accuracy_score(a.val,b)

class GNB():
    def __init__(self, bounds=None, epsilon=None):
        self.dgt = models.GaussianNB(bounds=bounds, epsilon=epsilon)

    def predict(self,x):
        return self.dgt.predict(x.val)

    def fit(self,x,y):
        # s = DataSource(genid(str(x)))
        s = 'source'
        add_privacy_cost(self.dgt.epsilon, 0, SensEnv({s:1}))
        return self.dgt.fit(x.val,y.val)


class LogisticRegression():
    def __init__(self, epsilon=1.0,data_norm=False):
        self.epsilon = epsilon
        if data_norm:
            self.dgt = models.LogisticRegression(epsilon=epsilon,data_norm=data_norm)
        else:
            self.dgt = models.LogisticRegression(epsilon=epsilon)

    def fit(self,x,y):
        # s = DataSource(genid(str(x)))
        s = 'source'
        add_privacy_cost(self.dgt.epsilon, 0, SensEnv({s:1}))
        return self.dgt.fit(x,y)

    def score(self,X,y):
        senv = SensEnv({'source': 1})
        return DuetWrapper(self.dgt.score(X,y), senv, LInf())
