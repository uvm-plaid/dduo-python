import sys
sys.path.append("../")
import duet

# create a dummy data source
source = duet.DataSource('dummy source')

# you can wrap any value in a DuetWrapper
val = duet.DuetWrapper(5, duet.SensEnv({source: 1.0}), duet.L1())

# printing a DuetWrapper shows the class of the wrapped value and its sensitivity env
print(val)

# doesn't change the sense env
print(f'val + 5: {val + 5}')

# doubles the sensitivity
print(f'val + val: {val + val}')

# sensitivity times 5
print(f'val * 5: {val * 5}')

# infinite sensitivity
print(f'val * val: {val * val}')

noisy_val = duet.renyi_gauss(val + val, α = 1.0, ε = 0.001)

print(f'noisy val: {noisy_val}')

from random import random
if random() < 0.5:
    x = duet.renyi_gauss(val + val, α = 1.0, ε = 0.001)
else:
    x = val + val

print(x)

duet.print_privacy_cost()
