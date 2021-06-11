from duet import sklearn as sk
from duet import diffprivlib as dpl
import diffprivlib.models as models
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import duet


dataset = sk.load_iris()
X_train, X_test, y_train, y_test = sk.train_test_split(dataset.data, dataset.target, test_size=0.2)

epsilons = np.logspace(-2, 2, 50)
bounds = [(4.3, 7.9), (2.0, 4.4), (1.1, 6.9), (0.1, 2.5)]
accuracy = list()
with duet.AdvEdOdometer() as odo:
    for epsilon in epsilons:
        clf = dpl.GNB(epsilon=epsilon)
        clf.fit(X_train, y_train)
        accuracy.append(dpl.score(y_test, clf.predict(X_test)))
    duet.print_privacy_cost()

# plt.semilogx(epsilons, accuracy)
# plt.title("Differentially private Naive Bayes accuracy")
# plt.xlabel("epsilon")
# plt.ylabel("Accuracy")
# plt.show()
