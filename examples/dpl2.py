import numpy as np
from duet import diffprivlib as dpl
from sklearn.linear_model import LogisticRegression
import sys

sys.path.append("../")
import duet

X_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                        usecols=(0, 4, 10, 11, 12), delimiter=", ")

y_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                        usecols=14, dtype=str, delimiter=", ")

X_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                        usecols=(0, 4, 10, 11, 12), delimiter=", ", skiprows=1)

y_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                        usecols=14, dtype=str, delimiter=", ", skiprows=1)
# Must trim trailing period "." from label
y_test = np.array([a[:-1] for a in y_test])

clf = LogisticRegression(solver="lbfgs")
clf.fit(X_train, y_train)

baseline = clf.score(X_test, y_test)
print("Non-private test accuracy: %.2f%%" % (baseline * 100))

thresh = 0.8
norm = 1200
delta = 0.001
noisy_acc = 0
epsilon = 0.01
step = 100

with duet.AdvEdOdometer() as odo:
    while noisy_acc < thresh:
        dp_clf = dpl.LogisticRegression(epsilon=epsilon,data_norm = norm)
        dp_clf.fit(X_train, y_train)
        accuracy = dp_clf.score(X_test, y_test)
        noisy_acc = duet.gauss(accuracy, ε = epsilon*2, δ = delta)
        norm -= step
        print(noisy_acc)

        duet.print_privacy_cost()

print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
     (dp_clf.epsilon, dp_clf.score(X_test, y_test) * 100))
