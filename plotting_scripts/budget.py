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

with open('op2.txt', 'r') as f:
    data = f.read()

data = data.split('\n')
data.pop()
data = [x.split('&')[-6] for x in data]
data = [x[3:-2] for x in data]

x = np.asarray([1,2,4,8,16,32,64,128,256])
start_from_8 = 3
means = [float(x.split('\pm')[0][:-1]) for x in data]
stds  = [float(x.split('\pm')[1][1:]) for x in data]
means = means[:-1]
print(means)
stds = np.asarray(stds[:-1]) / (590.2444458007812 + 386.9003601074219)

d = {}
d['Budget'] = np.tile(x, 4)
d['Value'] = np.asarray(means + [130.371, 150.749, 350.671, 417.117, 454.015, 454.544, 479.228, 496.620, 509.173] + [-65.341, -88.792, -68.321, -69.990, -89.296, -66.772, -115.439, -80.844, -91.687] + [129.271, 170.293, 355.372, 346.946, 504.366, 501.279, 522.161, 528.293, 533.187])
d['Value'] = (d['Value'] - (-386.9003601074219)) / (590.2444458007812 + 386.9003601074219)
d['Model'] = ["DDOM"] * len(means) + ["MINs"] * len(means) + ["Grad. Ascent"] * len(means) + ["COMs"] * len(means)

df = pd.DataFrame(d)
lineplot = sns.lineplot(data=df, x="Budget", y="Value", hue="Model", palette="colorblind", style='Model', markers=True, dashes=False)
fig = lineplot.get_figure()
c = lineplot.get_lines()
plt.fill_between(d['Budget'][:9], d['Value'][:9] - stds, d['Value'][:9] + stds, color=c[0].get_color(), alpha=0.3)
plt.fill_between(d['Budget'][9:18], d['Value'][9:18] - np.ones(9) * 0.016, d['Value'][9:18] + np.ones(9) * 0.016, color=c[1].get_color(), alpha=0.3)
plt.fill_between(d['Budget'][18:27], d['Value'][18:27] - np.ones(9) * 0.017, d['Value'][18:27] + np.ones(9) * 0.017, color=c[2].get_color(), alpha=0.3)
plt.fill_between(d['Budget'][27:], d['Value'][27:] - np.ones(9) * 0.017, d['Value'][27:] + np.ones(9) * 0.017, color=c[3].get_color(), alpha=0.3)
plt.xscale('log')
fig.savefig("budget.pdf", dpi=300)
"""
fig, ax = plt.subplots()
means = np.asarray(means)
stds = np.asarray(stds)
ax.plot(x, means, c=colors[0])
ax.fill_between(x, means-stds, means+stds, alpha=0.3, facecolor=colors[0])
ax.set_xscale('log')
ax.legend()
plt.savefig()
"""
