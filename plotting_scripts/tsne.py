import os
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

import design_bench

task = design_bench.make('DKittyMorphology-Exact-v0')
# task = design_bench.make('AntMorphology-Exact-v0')
# task = design_bench.make('Superconductor-RandomForest-v0')
task.map_normalize_x()
task.map_normalize_y()

oX = task.x[:5000]
oy = np.zeros(oX.shape[0])
qy = task.y[:5000].reshape(-1)

# base_path = f"./experiments/superconductor/cfd_gaussian_small_noise_reweighted_val/"
base_path = f"./experiments/dkitty/cfd_gaussian_small_noise_reweighted_val/"
# base_path = f"./experiments/ant/cfd_gaussian_small_noise_reweighted_val/"
base_expt_path = os.path.join(base_path, f"123/wandb/latest-run/files")
results_path = os.path.join(base_expt_path, f"results/latest-run/designs.pkl")

with open(results_path, 'rb') as f:
    nX = pkl.load(f)
    ny = np.ones(nX.shape[0])

X = np.concatenate([oX, nX], axis=0)
y = np.concatenate([oy, ny])
qy = np.concatenate([qy, 0 * ny])

feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
df['qy'] = qy

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = X.shape[0]
df_subset = df.loc[rndperm[:N], :].copy()
data_subset = df_subset[feat_cols].values
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))
s = sns.scatterplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    # hue="y",
    hue="qy",
    # palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="brief",
    palette="vlag",
    hue_norm=(-1, 1),
    alpha=0.3)
# fig = s.get_figure()
# fig.savefig('ant_tsne.png')
# plt.savefig('ant_tsne.png')
# fig.savefig('super_tsne.png')
# plt.savefig('super_tsne.png')
# fig.savefig('dkitty_tsne.png')
plt.savefig('dkitty_tsne.png')
