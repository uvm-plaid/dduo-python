import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 1.0
alpha = 1.0

df = pd.read_csv("test.csv")
print(df)

with duet.RenyiOdometer((alpha, 2.5)) as odo:
    for i in range(20):
        noisy_count = duet.renyi_gauss(df.shape[0], α = alpha, ε = epsilon)

    print(odo)

# print(f'result {r[0]}')
# print(f'odometer: {r[1]}')
