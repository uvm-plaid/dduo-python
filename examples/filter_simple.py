import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 1.0
alpha = 10

df = pd.read_csv("test.csv")

with duet.RenyiFilter(9,1.0):
    noisy_count = duet.renyi_gauss(df.shape[0], α = alpha, ε = epsilon)
    print(f'NoisyCount : {noisy_count}')

duet.print_privacy_cost()
