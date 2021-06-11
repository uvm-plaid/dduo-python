import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 0.0001

df = pd.read_csv("test.csv")
print(df.shape[0].val)
for i in range(20):
    noisy_count = duet.laplace(df.shape[0],ε=epsilon)
    print(f'Count {i}: {noisy_count}')

duet.print_privacy_cost()
