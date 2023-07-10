import torch
import sys
import os
from pprint import pformat
from pprint import pprint

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

# sys.path.append(os.path.join(os.getcwd()))
import torch
from botorch.test_functions.synthetic import Branin

def neg_branin(x):
    return -Branin().evaluate_true(torch.Tensor(x))

# from utils.functions import neg_branin

from matplotlib.colors import LogNorm, PowerNorm, Normalize, ListedColormap

seaborn_color_palette = sns.color_palette('bright').as_hex()
cmap = ListedColormap(seaborn_color_palette)

def plot_branin():
    fig, ax = plt.subplots()

    x1_values = np.linspace(-5, 10, 100)
    x2_values = np.linspace(0, 15, 100)
    x_ax, y_ax = np.meshgrid(x1_values, x2_values)
    vals = np.c_[x_ax.ravel(), y_ax.ravel()]
    fx = np.reshape([neg_branin(val) for val in vals], (100, 100))

    # cm = ax.pcolormesh(x_ax, y_ax, fx,
    # gamma = 0.4
    gamma = 2
    cm = ax.contour(x_ax, y_ax, fx, levels=np.arange(-300, -0.39, 5),
                       norm=PowerNorm(gamma=gamma, vmin=fx.min(),
                                    vmax=fx.max()),
                       cmap='rainbow')
                       # cmap='tab10_r')
                       # cmap='cividis_r')
                       # cmap=cmap)

    minima = np.array([[-np.pi, 12.275], [+np.pi, 2.275], [9.42478, 2.475]])
    ax.plot(minima[:, 0], minima[:, 1], "g.", markersize=8,
            lw=0, label="Maxima")

    mean = np.asarray([2.5, 7.5])
    std = np.asarray([4.33, 4.33])
    start = (minima[0] - mean) / std
    for 
    # traj = traj_pts[0].cpu().numpy()
    traj = np.asarray(eval_output.points)
    init_traj_0 = traj_pts[1].cpu().numpy()
    init_traj = init_traj_0[:32]
    ntraj = np.concatenate((init_traj, traj), axis=0)

    u = ntraj[::5]
    # ax.plot(farzi_pts[99,::5,0], farzi_pts[99,::5,1], 'r^--', label="Dataset")
    qq = farzi_pts2[3,::5]
    qq[-1,0] = 2
    qq[-1,1] = 1
    ax.plot(farzi_pts2[3,::5,0], farzi_pts2[3,::5,1], 'ko--', label="Dataset")
    # ax.plot(farzi_pts[1,::5,0], farzi_pts[1,::5,1], 'g^--')
    ax.plot(farzi_pts[2,::5,0], farzi_pts[2,::5,1], 'mo--', label="Dataset")
    ax.plot(u[:, 0], u[:, 1], '*b-', label="Model")
    ax.plot(u[-6:, 0], u[-6:, 1], '*r')
    # ax.annotate("", xy=((u[3,0] + u[4,0])/2, (u[3,1] + u[4,1])/2), xytext=((u[3,0] + u[3,0])/2, (u[3,1] + u[3,1])/2), arrowprops=dict(arrowstyle="-|>"))

    cb = fig.colorbar(cm)

    # ax.legend(loc="best", numpoints=1)
    ax.legend(loc='lower right', numpoints=1, fontsize=12)

    ax.set_xlabel("$X_0$", fontsize=16)
    ax.set_xlim([-5, 10])
    ax.set_ylabel("$X_1$", fontsize=16)
    ax.set_ylim([0, 15])

    # plt.show()
    plt.savefig('plots/branin/branin_traj_plot_fin_test.png')

plt.style.use('seaborn-colorblind')
plot_branin()
