import sys
sys.path.append("../")
import duet
from duet import pandas as pd
from duet import numpy as np
from duet import L2
from duet import LInf
from duet import DuetWrapper

adult = pd.read_csv('../data_long/adult_with_pii.csv')
age_counts = adult['Age'].value_counts().to_dict()
alpha = 10

def range_query(s, a, b):
    return np.sum([s.get(i,0) for i in range(a, b)])

syn_data = {k: 800 for k, v in age_counts.val.items()}
query_lowers = [int(np.random.uniform(18, 91)) for i in range(50)]
query_workload = [(lower, int(np.random.uniform(lower, 91))) for lower in query_lowers]
epsilon = 1.1
total_epsilon = 0.0

def mwem_update(k, x, lower, upper, real_ans, syn_ans, total):
    if (k >= lower and k <= upper):
        # if the query touches this row, update it using multiplicative weights rule
        return x * np.exp((real_ans - syn_ans)/(2*total))
    else:
        # otherwise don't update it
        return x

def mwem_step(e, real_data, syn_data):
    global total_epsilon
    # this only works for "range queries"
    (query,real_answer) = e
    lower, upper = query
    # calculate the "total weight" of the synthetic data
    total = np.sum([v for k, v in syn_data.items()])
    # calculate the "measurement" for the query
    duet.renyi_gauss(range_query(real_data, lower, upper), α = alpha, ε = epsilon)
    total_epsilon += epsilon
    syn_answer = range_query(syn_data, lower, upper)
    # update each row of the synthetic data using the multiplicative weights rule
    return dict([(k, mwem_update(k, x, lower, upper, real_answer, syn_answer, total)) for k, x in syn_data.items()])

curr_syn = syn_data

@duet.mode_switch(LInf,L2)
def L2_clip(v, b):
    v = duet.list2(list(map(lambda x : duet.unwrap(x),v)))
    norm = np.linalg.norm(v, ord=2)
    if norm.val > b:
        return b * (v / norm)
    else:
        return v

def err(real,syn):
    elist = []
    z = syn.items()
    for k,v in z:
        elist.append(np.abs(curr_syn[k] - real[k])/np.abs(real[k]))
    x = duet.list(elist)
    s = np.sum2(L2_clip(x,20))
    return s

# the outer loop of the MWEM algorithm
stable = 0
thresh = 37.0
stability_thresh = 1
iterations = 10

def expn(qw,data):
    l = []
    # mx = 0
    # mi = 0
    # i = 0
    for x in qw:
        (lower,upper) = x
        l.append(range_query(data, lower, upper))
    pl = duet.renyi_gauss_vec(duet.list2(l), α = alpha, ε = epsilon)
    m = max(pl)
    i = np.where3(pl == m)
    # i = pl.index(m)
    # print(i[0][0])

    return (qw[i[0][0]],m)

with duet.RenyiFilter(alpha,200.0):
    with duet.RenyiOdometer((alpha,2.1)) as odo:
        # while stable < stability_thresh:
        #     e = err(age_counts,curr_syn)
        #     print(f'actual: {e.val}')
        #     curr_noisy_err = duet.renyi_gauss2(alpha,2.1,e)
        #     total_epsilon += 1.0
        #     print(f'noisy: {curr_noisy_err}')
        #     if (curr_noisy_err < thresh):
        #         stable += 1
        #     else:
        #         stable = 0
        # print(query_workload)
        # print(age_counts)
        for t in range(iterations):
            e = expn(query_workload,age_counts.val)
            # for q in query_workload:
            curr_syn = mwem_step(e, age_counts, curr_syn)
        print(odo)
        print(err(age_counts,curr_syn).val)


print(f'Total budget used: {total_epsilon}')
