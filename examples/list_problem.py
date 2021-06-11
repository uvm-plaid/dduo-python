import sys
sys.path.append("../")
import duet
from duet import map

source = duet.DataSource('dummy source')

# this means 'ls' has LInf sensitivity of 1
ls = duet.DuetWrapper([1,2,3,4,5], duet.SensEnv({source: 1.0}), duet.LInf())

# **************************************************
# Test case 1: wrapped lists keep their sensitivities
# **************************************************
print("Test case 1")
print(ls)

# **************************************************
# Test case 2: list comprehension doesn't wrap the resulting list
# **************************************************
# but ls_p is a list of DuetWrappers, each with sensitivity 1
# this is semantically equivalent to map id ls
# so it *should* result in a list with L1 sensitivity 1
ls_p1 = map(lambda x: x, ls)
print("Test case 2")
print(ls_p1)


# **************************************************
# Test case 3: generator should wrap the resulting list and track sensitivity
# **************************************************
ls_p3 = map(lambda x: x+x, ls)

print("Test case 3")
print(ls_p3)

# **************************************************
# Test case 4: generator should detect a "non-map"
# **************************************************
ls_p4 = map(lambda x: ls[0], ls)

print("Test case 4")
print(ls_p4)

# **************************************************
# Test case 5: something bad should not happen when there is a side effect
# **************************************************
y = 1

def f(x):
    global y
    y = y + x
    return x

ls_p5 = map(lambda x: f(x), ls)
print("Test case 5")
print(ls_p5)
print(y)
