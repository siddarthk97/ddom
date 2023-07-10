#!/usr/bin/env python3

import os
import sys

import seaborn as sns
import pickle as pkl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

sns.set()
sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor': 'black'})
sns.set_context('paper')
sns.axes_style('ticks')
colors = sns.color_palette('colorblind').as_hex()

d = {'$N_B = 1$': np.asarray([77.310, 75.376, 71.381]), '$N_B = 32$': np.asarray([84.349, 107.993, 86.330]), '$N_B = 64$': np.asarray([88.396, 95.153, 84.397])}
# d += {'$K = 0.01N$': np.asarray([87.310, 95.376,101.381]), '$K = 0$': np.asarray([87.349, 110.993,87.330]), '$K = N$': np.asarray([87.396,94.153,85.397])}
FS = 16
tasks = ("Supercon.")

ind = np.arange(2)  # the x locations for the groups
width = 0.25       # the width of the bar

# sort_means_k = [103.600, 93.432, 91.427]
sort_means_k = [480.804, 531.641, 548.227]

# sort_stds_k = [12.62, 15.30, 10.26]
sort_stds_k = [8.139, 19.36, 15.30]

fig, ax = plt.subplots()
rects = []
for i in range(3):
    rects.append(ax.bar(0+i*width, sort_means_k[i], width, color=colors[i], yerr=sort_stds_k[i]))
# rects1 = ax.bar(0, sort_means_k[0], width, color=colors[0], yerr=sort_stds[0])
# for i in range(3,6):
#     rects.append(ax.bar(1+i*width, sort_means_t[i-3], width, color=colors[i], yerr=sort_stds_t[i-3]))
#
# rects2 = ax.bar(ind + width, sort_means[1], width, color=colors[1], yerr=sort_stds[1])

# add some text for labels, title and axes ticks
ax.set_ylabel('Best Function Value', fontsize=FS)
plt.xlabel("Ant", fontsize=FS)
plt.ylim([0, 560])
plt.yticks(fontsize=FS - 4)

# ax.legend((rects1[0], rects2[0]), ("Update", 'No update'), fontsize=FS-5, loc="upper left")
# ax.legend((x[0] for x in rects), ("$K=0.01N$", "$K=0.03N$", "$K=0.1N$"), fontsize=FS-5, loc="best")# , r"$\tau =R_{50}$", r"$\tau =R_{25}$", r"$\tau =R_{10}$"), fontsize=FS-5, loc="best")
ax.legend((x[0] for x in rects), (r"$N_B = 1$", r"$N_B = 32$", r"$N_B = 64$"), fontsize=FS-5, loc="best")# , ), fontsize=FS-5, loc="best")


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        print("height", height)
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                str(round(height, 2)),
                ha='center', va='bottom', fontsize=FS-4)

for r in rects:
    autolabel(r)
# plt.tight_layout()
plt.savefig("nb.pdf")
