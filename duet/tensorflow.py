from .duet_core import *
from .duet_envs import *

import tensorflow as tf
from tensorflow import train
from tensorflow import keras
from tensorflow import losses
# from tensorflow import config

import random
import string

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def genid(s):
    return abs(hash(s)) % (10 ** 8)

class DS():
    def __init__(self, l):
        super(DS, self).__init__()
        self.dgt = tf.keras.Sequential(l)

    def compile(self,optimizer=None, loss=None, metrics=['accuracy']):
        return self.dgt.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def evaluate(self,X,y,N):
        senv = SensEnv({'source': 1})
        return DuetWrapper(self.dgt.evaluate(X,y,N)[1], senv, LInf())

    def fit(self,eps,train_data, train_labels,epochs=None,validation_data=None,batch_size=None):
        # s = DataSource(genid(str(train_data)))

        add_privacy_cost(eps, 0, SensEnv({'source':1}))
        return self.dgt.fit(train_data, train_labels,epochs=epochs,validation_data=validation_data,batch_size=batch_size)
