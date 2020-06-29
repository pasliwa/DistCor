import numpy as np
import random
import timeit
import time
from functools import partial
from pathlib import Path
import itertools
import sys
import DistCor
import RegDep
import generate_network
from sklearn import linear_model


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


def time_backs_dist_cor(pair, cond, data_fr, Reps=500):
    s_test = partial(DistCor.test_partial_dCov_cond_Z, x = list(pair[0]),
                     y = list(pair[1]), z = cond, data = data_fr, Reps = Reps)
    t = timeit.Timer(s_test)
    time_, (test_stat, backs) = t.timeit(number=1)
    return time_, test_stat, backs


def time_backs_partial_cor(pair, cond, data_fr, Reps=500):
    s_test = partial(RegDep.test_partial_correlation_cond_Z, x = list(pair[0]),
                     y = list(pair[1]), z = cond, data = data_fr, Reps = Reps)
    t = timeit.Timer(s_test)
    time_, (test_stat, backs) = t.timeit(number=1)
    return time_, test_stat, backs


def time_backs_distance_resid_cor(pair, cond, data_fr, Reps=500):
    s_test = partial(RegDep.test_distance_resid_correlation_cond_Z, x = list(pair[0]),
                     y = list(pair[1]), z = cond, data = data_fr, model=linear_model.LassoLarsCV(normalize=False), Reps = Reps)
    t = timeit.Timer(s_test)
    time_, (test_stat, backs) = t.timeit(number=1)
    return time_, test_stat, backs


seed_val = 42
Reps = int(sys.argv[1])
min_conditioning_size = int(sys.argv[2])
max_conditioning_size = int(sys.argv[3])
file_name = sys.argv[4]
where_to_dir = sys.argv[5]  # "/media/piotrek/Seagate_Expansion_Drive/backgrounds_with_time/"

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
data_fr.to_csv(where_to_dir + f"data_for_{file_name.split('.')[0]}.csv")


for pair in findsubsets(data_fr.columns, 2):
    print(file_name, pair)
    for i in range(min_conditioning_size, max_conditioning_size):
        possible_conditionings = sorted(list(set(data_fr.columns) - set(pair)))
        conditioning_subsets = findsubsets(possible_conditionings, i)
        if len(conditioning_subsets) == 0:
            continue
        times = []
        test_stats = []
        backgrounds = []
        for cond in conditioning_subsets:
            t_, test_stat, backs = time_backs_dist_cor(pair, cond, data_fr, Reps=Reps)
            times.append(t_)
            test_stats.append(test_stat)
            backgrounds.append(backs)

######### SAVING RESULTS ###############
        saving_prefix = file_name.split(".")[0] + "_" + "-".join(pair) + "_" + str(i) + "_DistCor"

        saving_name = saving_prefix + "_BACKGROUNDS.npy"
        with open(where_to_dir + saving_name, "wb+") as backgrounds_file:
            np.save(backgrounds_file, np.array(backgrounds))

        saving_name = saving_prefix + "_TIMES.npy"
        with open(where_to_dir + saving_name, "wb+") as times_file:
            np.save(times_file, np.array(times))

        saving_name = saving_prefix + "_STATS.npy"
        with open(where_to_dir + saving_name, "wb+") as stats_file:
            np.save(stats_file, np.array(test_stats))

        saving_name = saving_prefix + "_SUBSETS.txt"
        with open(where_to_dir + saving_name, "w+") as subsets_file:
            save_text = ""
            for subs in conditioning_subsets:
                save_text += ",".join(subs) + "\n"
            print(save_text, file=subsets_file)

        times = []
        test_stats = []
        backgrounds = []
        for cond in conditioning_subsets:
            t_, test_stat, backs = time_backs_partial_cor(pair, cond, data_fr, Reps=Reps)
            times.append(t_)
            test_stats.append(test_stat)
            backgrounds.append(backs)

######### SAVING RESULTS ###############
        saving_prefix = file_name.split(".")[0] + "_" + "-".join(pair) + "_" + str(i) + "_partial_Cor"

        saving_name = saving_prefix + "_BACKGROUNDS.npy"
        with open(where_to_dir + saving_name, "wb+") as backgrounds_file:
            np.save(backgrounds_file, np.array(backgrounds))

        saving_name = saving_prefix + "_TIMES.npy"
        with open(where_to_dir + saving_name, "wb+") as times_file:
            np.save(times_file, np.array(times))

        saving_name = saving_prefix + "_STATS.npy"
        with open(where_to_dir + saving_name, "wb+") as stats_file:
            np.save(stats_file, np.array(test_stats))

        saving_name = saving_prefix + "_SUBSETS.txt"
        with open(where_to_dir + saving_name, "w+") as subsets_file:
            save_text = ""
            for subs in conditioning_subsets:
                save_text += ",".join(subs) + "\n"
            print(save_text, file=subsets_file)

        times = []
        test_stats = []
        backgrounds = []
        for cond in conditioning_subsets:
            t_, test_stat, backs = time_backs_distance_resid_cor(pair, cond, data_fr, Reps=Reps)
            times.append(t_)
            test_stats.append(test_stat)
            backgrounds.append(backs)

######### SAVING RESULTS ###############
        saving_prefix = file_name.split(".")[0] + "_" + "-".join(pair) + "_" + str(i) + "_DistResid"

        saving_name = saving_prefix + "_BACKGROUNDS.npy"
        with open(where_to_dir + saving_name, "wb+") as backgrounds_file:
            np.save(backgrounds_file, np.array(backgrounds))

        saving_name = saving_prefix + "_TIMES.npy"
        with open(where_to_dir + saving_name, "wb+") as times_file:
            np.save(times_file, np.array(times))

        saving_name = saving_prefix + "_STATS.npy"
        with open(where_to_dir + saving_name, "wb+") as stats_file:
            np.save(stats_file, np.array(test_stats))

        saving_name = saving_prefix + "_SUBSETS.txt"
        with open(where_to_dir + saving_name, "w+") as subsets_file:
            save_text = ""
            for subs in conditioning_subsets:
                save_text += ",".join(subs) + "\n"
            print(save_text, file=subsets_file)
