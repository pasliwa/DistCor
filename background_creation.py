import numpy as np
import pandas as pd
import random
import timeit
import time
from functools import partial
from pathlib import Path
import itertools
import sys
import DistCor
import generate_network


timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""



def findsubsets(s, n):
    return list(map(list, itertools.combinations(s, n)))

def time_backs(pair, cond, data_fr, Reps=500):
    indep_test = partial(DistCor.test_independence_cond_Z, x = list(pair[0]), y = list(pair[1]), z = cond, data = data_fr, Reps = Reps)
    t = timeit.Timer(indep_test)
    time_, (test_stat, backs) = t.timeit(number=1)
    return time_, test_stat, backs


seed_val = 42
Reps = int(sys.argv[1])
nsamp = int(sys.argv[2])
sigma = float(sys.argv[3])
min_conditioning_size = int(sys.argv[4])
max_conditioning_size = int(sys.argv[5])
where_to_dir = sys.argv[6] #"/media/piotrek/Seagate_Expansion_Drive/backgrounds_with_time/"
file_name = sys.argv[7]


Path(where_to_dir).mkdir(parents=True, exist_ok=True)

random.seed(seed_val)
np.random.seed(seed_val)

data_fr, Bayesian_network, noise_and_generating = generate_network.create_network_from_file(file_name)
dot = Bayesian_network.draw()
dot.name = file_name.split('.')[0]
dot.directory = where_to_dir
dot.format = 'png'
dot.filename = file_name.split(".")[0] 
dot.render()
data_fr.to_csv(where_to_dir + f"data_for_{file_name}_{nsamp}_{sigma}.csv")
independencies = Bayesian_network.get_all_independence_relationships()

indep_li = []
for indep in independencies:
    letters = sorted(list(indep[2]))
    indep_li.append("-".join([indep[0], indep[1]]) + "_" + "-".join(letters) + "\n")
sorted_indep_list = sorted(indep_li)
with open(where_to_dir + f"{file_name.split('.')[0]}_d-sep_independencies.txt", "w+") as out:
    print("".join(sorted_indep_list), file=out)

for pair in findsubsets(data_fr.columns, 2):


    for i in range(min_conditioning_size, max_conditioning_size):
        times = []
        test_stats = []
        backgrounds = []
        conditioning_subsets = findsubsets(data_fr.columns, i)
        for cond in conditioning_subsets:
            t_, test_stat, backs = time_backs(pair, cond, data_fr, Reps=Reps)
            times.append(t_)
            test_stats.append(test_stat)
            backgrounds.append(backs)

######### SAVING RESULTS ###############
        saving_name = "-".join(pair) + "_" + str(i) + "_" + str(sigma) + "_" + str(nsamp) + "_BACKGROUNDS.npy"
        with open(where_to_dir + saving_name, "wb+") as backgrounds_file:
            np.save(backgrounds_file, np.array(backgrounds))
        saving_name = "-".join(pair) + "_" + str(i) + "_" + str(sigma) + "_" + str(nsamp) + "_TIMES.npy"
        with open(where_to_dir + saving_name, "wb+") as times_file:
            np.save(times_file, np.array(times))
        saving_name = "-".join(pair) + "_" + str(i) + "_" + str(sigma) + "_" + str(nsamp) + "_STATS.npy"
        with open(where_to_dir + saving_name, "wb+") as stats_file:
            np.save(stats_file, np.array(test_stats))
        saving_name = "-".join(pair) + "_" + str(i) + "_" + str(sigma) + "_" + str(nsamp) + "_SUBSETS.txt"
        with open(where_to_dir + saving_name, "w+") as subsets_file:
            save_text = ""
            for subs in conditioning_subsets:
                save_text += ",".join(subs) + "\n"
            print(save_text, file=subsets_file)


