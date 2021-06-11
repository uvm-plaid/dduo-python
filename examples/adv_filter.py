import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 0.01
delta = 0.001

df = pd.read_csv("test.csv")

duet.accountant.register_advod(duet.AdvEdOdometer())
noisy_count = 0
iters = 100
f = duet.AdvEDFilterObj(10,10)
with f:
    for i in range(iters):
        noisy_count += duet.gauss(df.shape[0], ε=epsilon, δ=delta)
print(f'NoisyCount : {noisy_count/iters}')

duet.print_privacy_cost()
