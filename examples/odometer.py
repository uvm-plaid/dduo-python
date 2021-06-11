import duet

from duet import pandas as pd
adult_data = pd.read_csv("adult_with_pii.csv")

def dp_counting_query(df, col, val, epsilon):
    val = df[df[col] == val].shape[0]
    print(val)
    return duet.laplace(val, ε=epsilon)

epsilon = 0.01
delta = 0.00001
with duet.EDOdometer() as odo:
    print('Query result:', dp_counting_query(adult_data, 'Marital Status', 'Never-married', epsilon))
    print('Privacy cost:', odo)
