import json
import os
import configargparse

from typing import Optional, Union
from pprint import pprint

import numpy as np
import pandas as pd
import pickle as pkl

from util import TASKNAME2TASK

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        description="Reinforcement Learning via Supervised Learning",)
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys()),
                        nargs='+',
                        required=True)
    parser.add_argument("--name", type=str, nargs='+', help="Experiment name")

    args = parser.parse_args()

    seeds = [123, 234, 345, 456, 567]

    final_means = [[None for t in args.task] for n in args.name]
    final_std = [[None for t in args.task] for n in args.name]
    for task in args.task:
        for name in args.name:
            data = []
            for seed in seeds:
                expt_save_path = f"./experiments/{task}/{name}/{seed}"
                assert os.path.exists(expt_save_path)

                save_results_dir = os.path.join(
                    expt_save_path, "wandb/latest-run/files/results/")
                with open(os.path.join(save_results_dir, 'results.pkl'),
                          'rb') as f:
                    data.append(pkl.load(f))

            data = np.asarray(data)
            print(data.shape)
