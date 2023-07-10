"""Utilities used throughout the codebase."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from wandb.sdk.wandb_run import Run

TASKNAME2TASK = {
    'dkitty': 'DKittyMorphology-Exact-v0',
    'ant': 'AntMorphology-Exact-v0',
    'tf-bind-8': 'TFBind8-Exact-v0',
    'tf-bind-10': 'TFBind10-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
    # 'hopper': 'HopperController-Exact-v0',
    'nas': 'CIFARNAS-Exact-v0',
    'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
    # 'gfp': 'GFP-Transformer-v0',
}


def configure_gpu(use_gpu: bool, which_gpu: int) -> torch.device:
    """Set the GPU to be used for training."""
    if use_gpu:
        device = torch.device("cuda")
        # Only occupy one GPU, as in https://stackoverflow.com/questions/37893755/
        # tensorflow-set-cuda-visible-devices-within-jupyter
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return device


def set_seed(seed: Optional[int]) -> None:
    """Set the numpy, random, and torch random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def sorted_glob(*args, **kwargs) -> List[str]:
    """A sorted version of glob, to ensure determinism and prevent bugs."""
    return sorted(glob.glob(*args, **kwargs))


def parse_val_loss(filename: str) -> float:
    """Parse val_loss from the checkpoint filename."""
    start = filename.index("val_loss=") + len("val_loss=")
    try:
        end = filename.index("-v1.ckpt")
    except ValueError:
        end = filename.index(".ckpt")
    val_loss = float(filename[start:end])
    return val_loss


## REWEIGHTING
def adaptive_temp_v2(scores_np):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    quantile_ninety = np.quantile(scores_new, q=0.9)
    return np.maximum(np.abs(quantile_ninety), 0.001)


def softmax(arr, temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores, base_temp=None):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    if base_temp is None:
        base_temp = adaptive_temp_v2(scores_np)
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)
    print(provable_dist)

    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, 19)]

    weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float32)[:, np.newaxis]
