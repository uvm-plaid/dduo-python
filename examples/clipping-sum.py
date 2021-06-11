import sys
sys.path.append("../")
import duet
from duet import pandas as pd

epsilon = 1.0
delta = 0.00001

df = pd.read_csv("test.csv")
clipped_df = df['salary'].clip(50000, 70000)

noisy_sum = duet.laplace(clipped_df.sum(), epsilon=epsilon)
noisy_count = duet.laplace(df.shape[0], epsilon=epsilon)

print(noisy_sum / noisy_count)

duet.print_privacy_cost()
