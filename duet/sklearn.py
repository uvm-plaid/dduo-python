from .duet_core import *
from .duet_envs import *
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split as tts

dataset = ds.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

def load_iris():
    return DuetWrapper(dataset,SensEnv({DataSource('iris'):1}),LInf())


def train_test_split(d,t,test_size=0.2):
    X_train, X_test, y_train, y_test = tts(d.val, t.val, test_size=0.2)
    a = DuetWrapper(X_train,d.senv,LInf())
    b = DuetWrapper(X_test,d.senv,LInf())
    c = DuetWrapper(y_train,d.senv,LInf())
    d = DuetWrapper(y_test,d.senv,LInf())
    return (a,b,c,d)


# load_iris = load_iris
# train_test_split = train_test_split
