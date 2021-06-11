import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 1.0
alpha = 10

df = pd.read_csv("test.csv")

with duet.RenyiOdometer((10,1.0)) as odo:
    f1, f2 = duet.RenyiFilterObj(10, 2.0).split()
    with f1:
        noisy_count1 = duet.renyi_gauss(df.shape[0], α = alpha, ε = epsilon)
    with f2:
        noisy_count2 = duet.renyi_gauss(df.shape[0], α = alpha, ε = epsilon)

    print('Privacy cost:', odo)
