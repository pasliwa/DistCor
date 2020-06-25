import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import time
from functools import partial
from pathlib import Path
import sys

def read_in_data(director, model_name, pair, conditioning_size, methods):
    saving_prefix = director + model_name + "_" + "-".join(pair) + "_" + str(conditioning_size)
    data = {}
    for method in methods:
        test_stats = np.load(f"{saving_prefix}_{method}_STATS.npy")
        times = np.load(f"{saving_prefix}_{method}_TIMES.npy")
        backgrounds = np.load(f"{saving_prefix}_{method}_BACKGROUNDS.npy")
        with open(f"{saving_prefix}_{method}_SUBSETS.txt", "r") as hand:
            subsets = list(map(str.strip, hand.readlines()))
        if len(subsets) > 1 and subsets[-1] == "":
            subsets = subsets[:-1]
        data[method] = (test_stats, times, backgrounds, subsets)
    return data

seed_val = 42
model_name = sys.argv[1]
where_to_dir = sys.argv[2]  # "/media/piotrek/Seagate_Expansion_Drive/backgrounds_with_time/"

threshold = float(sys.argv[3])
conditioning_size = int(sys.argv[4])
pair = sorted(sys.argv[5].split("-"))
where_to_plots = where_to_dir + "plots/"
Path(where_to_plots).mkdir(parents=True, exist_ok=True)

true_indep = []
with open(where_to_dir + f"{model_name}_d-sep_independencies.txt", "r") as hand:
    true_indep = list(map(str.strip, hand.readlines()))

methods = ["DistCor", "DistResid", "partial_Cor"]
data = read_in_data(where_to_dir, model_name, pair, conditioning_size, methods)

for method in methods:
    test_stats, times, backgrounds, subsets = data[method]

    sorted_data = sorted(zip(test_stats, backgrounds.tolist(), times.tolist(), subsets), reverse=True)
    for i in range(len(subsets)):
        d = [sorted_data[i] for i in range(len(subsets)) if ((pair[0] not in sorted_data[i][-1]) and (pair[1] not in sorted_data[i][-1]))]
    tstats, backs, tim, subs = zip(*d)

    if method == "DistCor":
        if conditioning_size == 0:
            method_desc = "Distance Covariance"
        else:
            method_desc = "Partial Distance Covariance"
    elif method == "DistResid":
        if conditioning_size == 0:
            method_desc = "Correlation of UCDM"
        else:
            method_desc = "Partial Correlation of UCDM"
    else:
        if conditioning_size == 0:
            method_desc = "Correlation"
        else:
            method_desc = "Partial Correlation"

    fig, axs = plt.subplots(len(tstats), sharex=True, gridspec_kw={'hspace': 0}, figsize=(10, 5 * len(tstats)))
    fig.suptitle(f'Test of {method_desc} = 0 for {pair[0]} and {pair[1]} controlling for {conditioning_size} other variable{"s" if conditioning_size > 1 else ""}')
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(tstats)):
        inde = "-".join([pair[0], pair[1]]) + "_" + "-".join(sorted(subs[i].replace(",", "")))
        true_cond_indep = False
        if (pair[0] in subs[i]) or (pair[1] in subs[i]) or (inde in true_indep):
            true_cond_indep = True

        if len(tstats) == 1:
            sns.distplot(backs[i], ax=axs, kde=False, bins=30)
            axs.plot(tstats[i], 0, "o")
            minimal, maximal = np.percentile(backs[i], threshold / 2), np.percentile(backs[i], 100 - (threshold / 2))
            line_style = "-" if true_cond_indep else ":"
            axs.axvline(minimal, ls=line_style)
            axs.axvline(maximal, ls=line_style)
            axs.set_ylabel(subs[i], rotation=0, labelpad=12 + len(subs[i]))
            axs.set_yticks([])
            estimated_independence = True
            if (tstats[i] < minimal) or (maximal < tstats[i]):
                estimated_independence = False
            if true_cond_indep != estimated_independence:
                axs.set_facecolor("#FFCCCC")
            if true_cond_indep == estimated_independence and estimated_independence == True:
                true_pos += 1
            if true_cond_indep == estimated_independence and estimated_independence == False:
                true_neg += 1
            if true_cond_indep != estimated_independence and estimated_independence == True:
                false_pos += 1
            if true_cond_indep != estimated_independence and estimated_independence == False:
                false_neg += 1

        else:
            sns.distplot(backs[i], ax=axs[i], kde=False, bins=30)
            axs[i].plot(tstats[i], 0, "o")
            minimal, maximal = np.percentile(backs[i], threshold / 2), np.percentile(backs[i], 100 - (threshold / 2))
            line_style = "-" if true_cond_indep else ":"
            axs[i].axvline(minimal, ls=line_style)
            axs[i].axvline(maximal, ls=line_style)
            axs[i].set_ylabel(subs[i], rotation=0, labelpad=12 + len(subs[i]))
            axs[i].set_yticks([])
            estimated_independence = True
            if (tstats[i] < minimal) or (maximal < tstats[i]):
                estimated_independence = False
            if true_cond_indep != estimated_independence:
                axs[i].set_facecolor("#FFCCCC")
            if true_cond_indep == estimated_independence and estimated_independence == True:
                true_pos += 1
            if true_cond_indep == estimated_independence and estimated_independence == False:
                true_neg += 1
            if true_cond_indep != estimated_independence and estimated_independence == True:
                false_pos += 1
            if true_cond_indep != estimated_independence and estimated_independence == False:
                false_neg += 1

        	# Hide x labels and tick labels for all but bottom plot.
            for ax in axs:
                ax.label_outer()

        plt.savefig(where_to_plots + f"{model_name}_{method}_{pair[0]}_{pair[1]}_cond_{conditioning_size}.png")
        fig = plt.figure()

        plt.title(f'Test of {method_desc} = 0 for {pair[0]} and {pair[1]} controlling for {conditioning_size} other variable{"s" if conditioning_size > 1 else ""}')
        sns.heatmap(np.array([[true_pos, false_neg], [false_pos, true_neg]]), annot=True, fmt="d", cmap="Reds")
        plt.savefig(where_to_plots + f"{method}_{pair[0]}_{pair[1]}_cond_{conditioning_size}_confusion.png")
