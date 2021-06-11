import duet

# create a dummy data source
source = duet.DataSource('dummy source')

# you can wrap any value in a DuetWrapper
val = duet.DuetWrapper(5, duet.SensEnv({source: 1.0}), duet.L1())


print(duet.gauss(val, epsilon = 1.0, delta = 1e-5))
print(val == 5)

if val == 5:
    print(1)
else:
    print(2)

print(val)

duet.print_privacy_cost()
