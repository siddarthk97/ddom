import numpy as np
import pandas as pd
import pickle as pkl

import os
import configargparse

from util import TASKNAME2TASK, TASKNAME2FULL

from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
    from design_bench.datasets.discrete.gfp_dataset import GFPDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    parser.add_argument("--names",
                        type=str,
                        help="Experiment name",
                        nargs="+",
                        required=True)
    parser.add_argument('--seeds',
                        type=int,
                        nargs="+",
                        default=[123, 234, 345, 456, 567])
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--normalise", default=False, action='store_true')

    args = parser.parse_args()
    
    x = {'dkitty':0.926, 'ant':0.957, 'tf-bind-8':0.971, 'tf-bind-10':0.688, 'superconductor':0.560, 'nas':0, 'chembl':0.633}
    y = {'dkitty':0.009, 'ant':0.012, 'tf-bind-8':0.005, 'tf-bind-10':0.092, 'superconductor':0.044, 'nas':0, 'chembl':0.007}

    results = {t: list() for t in args.names}
    for expt in args.names:
        for task in TASKNAME2TASK.keys():
            res_per_task = []
            base_path = f"./experiments/{task}/{expt}/"
            if os.path.exists(base_path):
                for seed in args.seeds:
                    base_expt_path = os.path.join(
                        base_path, f"{seed}/wandb/latest-run/files")

                    results_path = os.path.join(
                        base_expt_path, f"results/latest-run/results.pkl")
                    if not os.path.exists(results_path):
                        results_path = os.path.join(base_expt_path,
                                                    f"results/results.pkl")
                        if not os.path.exists(results_path):
                            continue

                    with open(results_path, 'rb') as f:
                        result = pkl.load(f)

                    if args.normalise:
                        full_task = TASKNAME2FULL[task]
                        result = (result - full_task.y.min()) / (full_task.y.max() - full_task.y.min())
                    if args.budget == 0:
                        res_per_task.append(result)
                    else:
                        res_per_task.append(result[:args.budget, ...])

                results[expt].append(res_per_task)
            else:
                results[expt].append(None)

    raw_results_df = pd.DataFrame(results, index=list(TASKNAME2TASK.keys()))

    def aggregate_score(x):
        if x is None:
            return None

        res = []
        for r in x:
            res.append(r.max())

        res = np.asarray(res)
        return f"${res.mean():.3f} \pm {res.std():.3f}$"

    def avg_score(x):
        if x is None:
            return None

        res = []
        for r in x:
            # res.append(r.mean())
            res.append(np.median(r))

        res = np.asarray(res)
        return f"${res.mean():.3f} \pm {res.std():.3f}$"

    summary_df = raw_results_df.applymap(aggregate_score)
    # summary_df = raw_results_df.applymap(avg_score)
    summary_df = summary_df.transpose()
    print(summary_df.to_string())
    print("=" * 20)
    print(summary_df.to_latex(escape=False))
