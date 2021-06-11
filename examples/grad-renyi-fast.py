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

@duet.mode_switch(LInf,L2)
def L2_clip(v, b):
    norm = np.linalg.norm(v, ord=2)
    if norm.val > b:
        return b * (v / norm)
    else:
        return v

@duet.mode_switch(LInf,L2)
def L2_clip_array(vs, b):
    norms = np.linalg.norm(vs, ord=2, axis=1)
    ratios = vs / norms[:, None]
    results = np.where((norms > b)[:, None], b*ratios, vs)
    return results

def ff(a,b,theta):
    (x_i, y_i) = a
    return L2_clip(gradient(theta, x_i, y_i), b)

def vgradient(theta_in, x_in, y_in, b):
    x = x_in
    y = y_in
    theta = theta_in
    exponent = y * np.dot(x,theta)
    rhs = (y / (1+np.exp(exponent)))
    gradients = -(x*rhs[:, None])
    clipped_grads = L2_clip_array(gradients, b)
    return np.sum(clipped_grads, axis=0)

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
        grad_sum        = vgradient(theta, X_train, y_train, sensitivity)
        noisy_grad_sum  = gaussian_mech_vec(grad_sum, alpha, eps_i)
        noisy_avg_grad  = noisy_grad_sum / noisy_count
        theta           = np.subtract(theta, noisy_avg_grad)

    return theta

iterations = 100

with duet.RenyiDP(1e-5):
    theta = dp_gradient_descent(iterations, alpha, epsilon)
    print(theta.val)

acc = accuracy(theta)
print(f"Epsilon = {epsilon}, final accuracy: {acc.val}")
duet.print_privacy_cost()
