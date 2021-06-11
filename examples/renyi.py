import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 1.0
alpha = 10

df = pd.read_csv("test.csv")

with duet.RenyiDP(0.001) as odo:
    noisy_count = duet.renyi_gauss(df.shape[0], α = alpha, ε = epsilon)
    print(f'NoisyCount : {noisy_count}')
    print('Privacy cost:', odo)
