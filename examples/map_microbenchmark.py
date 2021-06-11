import sys
import duet
import time

source = duet.DataSource('dummy source')
LIST_SIZE = 1000000

ls = duet.DuetWrapper(list(range(LIST_SIZE)), duet.SensEnv({source: 1.0}), duet.LInf())
print(ls)

start = time.time()
result = duet.map(lambda x: x + 1, ls)
end = time.time()
elapsed = end - start
print('Private elapsed:', elapsed)

ls_nonprivate = list(range(LIST_SIZE))
start = time.time()
result = map(lambda x: x + 1, ls_nonprivate)
end = time.time()
elapsed = end - start
print('Non-Private elapsed:', elapsed)
