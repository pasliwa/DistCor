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

def read_in_data(director, pair, i, nsamp, sigma):
    joined_pairs = "-".join(pair)
    test_stats = np.load(f"{director}{joined_pairs}_{i}_{sigma}_{nsamp}_STATS.npy")
    times = np.load(f"{director}{joined_pairs}_{i}_{sigma}_{nsamp}_TIMES.npy")
    backgrounds = np.load(f"{director}{joined_pairs}_{i}_{sigma}_{nsamp}_BACKGROUNDS.npy")
    with open(f"{director}{joined_pairs}_{i}_{sigma}_{nsamp}_SUBSETS.txt", "r") as hand:
        subsets = list(map(str.strip, hand.readlines()))
    return test_stats, times, backgrounds, subsets


seed_val = 42
nsamp = int(sys.argv[1])
sigma = float(sys.argv[2])
threshold = float(sys.argv[3])
conditioning_size = int(sys.argv[4])
where_to_dir = sys.argv[5] #"/media/piotrek/Seagate_Expansion_Drive/backgrounds_with_time/"
file_name = sys.argv[6]
pair = sorted(sys.argv[7].split("-"))
where_to_plots = where_to_dir + "plots/"
Path(where_to_plots).mkdir(parents=True, exist_ok=True)

test_stats, times, backgrounds, subsets = read_in_data(where_to_dir, pair, conditioning_size, nsamp, sigma)

tstats, backs, tim, subs = zip(*sorted(zip(test_stats, backgrounds.tolist(), times.tolist(), subsets), reverse=True))

true_indep = []
with open(where_to_dir + f"{file_name.split('.')[0]}_d-sep_independencies.txt", "r") as hand:
    true_indep = list(map(str.strip, hand.readlines()))

fig, axs = plt.subplots(len(tstats), sharex=True, gridspec_kw={'hspace': 0}, figsize=(10, 25))
fig.suptitle(f'{pair[0]} conditionally dependent on {pair[1]} given {conditioning_size} variable{"s" if conditioning_size > 1 else ""}')
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i in range(len(tstats)):
    inde = "-".join([pair[0], pair[1]]) + "_" + "-".join(sorted(subs[i].replace(",", "")))
    true_cond_indep = False
    if (pair[0] in subs[i]) or (pair[1] in subs[i]) or (inde in true_indep):
        true_cond_indep = True

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

plt.savefig(where_to_plots + f"{pair[0]}_{pair[1]}_cond_{conditioning_size}.png")
fig = plt.figure()
plt.title(f'{pair[0]} conditionally dependent on {pair[1]} given {conditioning_size} variable{"s" if conditioning_size > 1 else ""}')
sns.heatmap(np.array([[true_pos, false_neg], [false_pos, true_neg]]), annot=True, fmt="d", cmap="Reds")
plt.savefig(where_to_plots + f"{pair[0]}_{pair[1]}_cond_{conditioning_size}_confusion.png")