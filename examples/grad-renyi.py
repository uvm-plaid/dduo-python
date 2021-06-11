import sys
sys.path.append("../")
import duet
from duet import pandas as pd
from duet import numpy as np
from duet import map
from duet import L2
from duet import LInf
from duet import zip

epsilon = 1.0
alpha = 10

X = np.load('../data_long/adult_processed_x.npy')
y = np.load('../data_long/adult_processed_y.npy')

training_size = int(X.shape[0] * 0.8)

X_train = X[:training_size]
X_test = X[training_size:]

y_train = y[:training_size]
y_test = y[training_size:]

def gaussian_mech_vec(v, alpha, epsilon):
    return duet.renyi_gauss_vec(v, α = alpha, ε = epsilon)

def gradient(theta, xi, yi):
    exponent = yi * np.dot(xi,theta)
    e = (1+np.exp(exponent))
    return - (yi*xi) / e

# TODO : this should probably be in duet_core
@duet.mode_switch(LInf,L2)
def L2_clip(v, b):
    norm = np.linalg.norm(v, ord=2)
    if norm.val > b:
        return b * (v / norm)
    else:
        return v

def unpack_and_clip(a,b,theta):
    (x_i, y_i) = a
    return L2_clip(gradient(theta, x_i, y_i), b)

def gradient_sum(theta, X, y, b):
    gradients = map(lambda x: unpack_and_clip(x,b,theta), zip(X,y))
    return np.sum(gradients, axis=0)

def accuracy(theta):
    return np.sum(predict(theta, X_test) == y_test)/X_test.shape[0]

def predict(theta, xi):
    label = np.sign(xi @ theta)
    return label

def dp_gradient_descent(iterations, alpha, eps):
    eps_i = eps/iterations
    theta = np.zeros(X_train.shape[1])
    sensitivity = 5
    noisy_count = duet.renyi_gauss(X_train.shape[0], α = alpha, ε = eps)
    for i in range(iterations):
        grad_sum        = gradient_sum(theta, X_train, y_train, sensitivity)
        noisy_grad_sum  = gaussian_mech_vec(grad_sum, alpha, eps_i)
        noisy_avg_grad  = noisy_grad_sum / noisy_count
        theta           = np.subtract(theta, noisy_avg_grad)

    return theta

iterations = 3

epses = [0.001]

for eps in epses:
    with duet.RenyiDP(1e-5):
        theta = dp_gradient_descent(iterations, alpha, eps)
        # print(theta.val)

    acc = accuracy(theta)
    print(f"Epsilon = {eps}, final accuracy: {acc.val}")
    duet.print_privacy_cost()
