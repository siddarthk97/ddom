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

# plt.rcParams['text.usetex'] = True

# task = "superconductor"
# task = "ant"
# task = "dkitty"
task = sys.argv[1]
name = "cfd_gaussian_small_noise_reweighted_val"
seeds = [123, 234, 345, 456, 567]
num_samples = 512
num_steps = 1000
# conditions = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
condition = 4.0
gammas = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0, 10.0]
gamma = 2.5
beta_min = 0.01
beta_max = 2.0

min_max = {
    "ant": [-386.9003601074219, 590.2444458007812],
    "dkitty": [-880.45849609375, 340.90985107421875],
    "superconductor": [0.0002099999983329326, 185.0],
}

cond_result_map = {}
for gamma in gammas:
    for seed in seeds:
        expt_save_path = f"./experiments/{task}/{name}/{seed}"
        save_results_dir = os.path.join(expt_save_path,
                                        f"wandb/latest-run/files/results/")

        directories = os.listdir(save_results_dir)
        # print(directories)

        run_specific_str_prefix = f"{num_samples}_{num_steps}_{condition}_{gamma}_{beta_min}_{beta_max}"
        x = [x for x in directories if run_specific_str_prefix in x]
        # print(condition, x)
        if len(x) != 0:
            res_path = os.path.join(save_results_dir, x[0], 'results.pkl')
            if gamma not in cond_result_map.keys():
                cond_result_map[gamma] = [res_path]
            else:
                cond_result_map[gamma].append(res_path)


def norm_pkl(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
        data = (data - min_max[task][0]) / (min_max[task][1] - min_max[task][0])

        return data


x = []
y = []
d = {'Condition': [], 'Seeds': [], 'Value': []}

for k, v in cond_result_map.items():
    vals = [norm_pkl(x) for x in v]

    x.append(k)
    y.append([[x.mean() for x in vals], [x.std() for x in vals]])
    d['Condition'] += [k] * len(vals) * 512
    d['Seeds'] += seeds * 512
    z = np.concatenate(vals, axis=0)
    z = z.reshape(-1)
    d['Value'] += z.tolist()

y_0 = [[x[0][0], x[1][0]] for x in y]
y_0 = np.asarray(y_0)
x = np.asarray(x).reshape(-1, 1)
df = pd.DataFrame(d)  #, index=[str(x) for x in conditions])
print(df)

data = np.concatenate((x, y_0), axis=1)

# lineplot = sns.lineplot(x=data[:,0], y=data[:,1])
lineplot = sns.lineplot(data=df,
                        x="Condition",
                        y="Value",
                        hue="Seeds",
                        palette="colorblind",
                        estimator=np.mean)
fig = lineplot.get_figure()
fig.savefig('test_gamma.pdf', dpi=300)
