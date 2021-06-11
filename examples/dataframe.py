import sys
sys.path.append("../")
import duet
from duet import pandas as pd

df = pd.read_csv("test.csv")
filtered_df = df[df['salary'] > 60000]

epsilon = 1.0
delta = 0.00001

noisy_answer = duet.gauss(filtered_df.shape[0] + filtered_df.shape[0], ε=epsilon, δ=delta)
noisy_answer = duet.gauss(filtered_df.shape[0], ε=epsilon, δ=delta)

print(noisy_answer)

duet.print_privacy_cost()
