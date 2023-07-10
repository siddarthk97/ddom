import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# sns.set('seaborn-colorblind')
sns.set()
sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
sns.set_context('paper')
sns.axes_style('ticks')
colors = sns.color_palette('colorblind').as_hex()

plt.rcParams['text.usetex'] = True

x = np.asarray([0.01, 0.5, 1, 2, 3, 5, 6])
y = np.asarray([491.549, 501.453, 508.750,  509.672, 462.584, 327.932,394.160,])
std = np.asarray([23.879, 20.905, 11.453,  33.464, 26.250,95.462,])

plt.plot(x, y)
plt.axvline(2, 0, 500)
plt.savefig('test.png')
