#!/usr/bin/env python3

import os
import sys

import seaborn as sns
import pickle as pkl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

sns.set()
sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
sns.set_context('paper')
sns.axes_style('ticks')
colors = sns.color_palette('colorblind').as_hex()

# plt.rcParams['text.usetex'] = True

task = "dkitty"
name = "cfd_hidden_size"
sizes = [32, 64, 128, 256, 512, 2048, 4096, 8192, 16384]
seeds = [123,]
num_samples = 512
num_steps = 1000
# conditions = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
conditions = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
gamma = 2.5
beta_min = 0.01
beta_max = 2.0

s_max = 340.
s_min = -880.

cond_result_map = {}
for size in sizes:
    for seed in seeds:
        expt_save_path = f"./experiments/{task}/{name}_{size}/{seed}"
        save_results_dir = os.path.join(expt_save_path, f"wandb/latest-run/files/results/latest-run")

        # directories = os.listdir(save_results_dir)
        # print(directories)

        # print(condition, x)
        res_path = os.path.join(save_results_dir, 'results.pkl')
        if size not in cond_result_map.keys():
            cond_result_map[size] = [res_path]
        else:
            cond_result_map[size].append(res_path)

def norm_pkl(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
        data = (data - s_min) / (s_max - s_min)

        return data
x = []
y = []
d = {'size': [], 'seeds': [], 'values': []}

for k, v in cond_result_map.items():
    vals = [norm_pkl(x) for x in v] 

    x.append(k)
    y.append([[x.mean() for x in vals], [x.std() for x in vals]])
    d['size'] += [k] * len(vals) * 512
    d['seeds'] += seeds * 512
    z = np.concatenate(vals, axis=0)
    z = z.reshape(-1)
    d['values'] += z.tolist()

y_0 = [[x[0][0], x[1][0]] for x in y]
y_0 = np.asarray(y_0)
x = np.asarray(x).reshape(-1, 1)
df = pd.DataFrame(d) #, index=[str(x) for x in conditions])
print(df)

data = np.concatenate((x, y_0), axis=1)

# lineplot = sns.lineplot(x=data[:,0], y=data[:,1])
lineplot = sns.lineplot(data=df, x="size", y="values", hue="seeds", palette="colorblind", estimator=np.max)
lineplot.set(xscale='log')
fig = lineplot.get_figure()
fig.savefig('test.png')
