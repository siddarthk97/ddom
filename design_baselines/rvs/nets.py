import sys
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Optional, Tuple, Type


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

import numpy as np
import pytorch_lightning as pl

import torch
from torch import optim, nn, utils, Tensor

from util import TASKNAME2TASK


class RvSContinuous(pl.LightningModule):

    def __init__(self,
                 taskname,
                 task,
                 hidden_size=1024,
                 learning_rate=1e-3,
                 activation_fn=nn.ReLU,
                 dropout_p=0.1,
                 w=None):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.p = dropout_p
        self.dim_y = self.task.y.shape[-1]
        self.dim_x = self.task.x.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(self.dim_y, hidden_size),
            activation_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, self.dim_x),
        )
        self.w = w

    def forward(self, y):
        y = y.view(y.size(0), -1)
        x_hat = self.model(y)

        return x_hat

    def training_step(self, batch, batch_idx, log_prefix="train"):
        x, y, w = batch
        # x, y = batch
        x_hat = self.forward(y)
        # w = None
        if w is not None:
            loss = (x_hat - x)**2
            loss = torch.mean(w * loss)
        else:
            loss = nn.functional.mse_loss(x_hat, x)

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class RvSDiscrete(pl.LightningModule):

    def __init__(self,
                 taskname,
                 task,
                 hidden_size=1024,
                 learning_rate=1e-3,
                 activation_fn=nn.ReLU,
                 dropout_p=0.1,
                 w=None):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.p = dropout_p
        self.dim_y = self.task.y.shape[-1]
        self.dim_x = self.task.x.shape[-1]
        self.model = nn.Sequential(
            nn.Linear(self.dim_y, hidden_size),
            activation_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, self.dim_x * self.task.num_classes),
        )
        self.w = w

    def forward(self, y):
        y = y.view(y.size(0), -1)
        x_hat = self.model(y)

        return x_hat

    def training_step(self, batch, batch_idx, log_prefix="train"):
        x, y, w = batch
        # x, y = batch
        x_hat = self.forward(y)
        # w = None
        x_hat = x_hat.view(x.size(0), self.dim_x, -1)
        w = None
        if w is not None:
            loss_acc = torch.zeros(x.size(0)).to(x.device)
            for i in range(x.size(1)):
                loss_acc = loss_acc + nn.functional.cross_entropy(x_hat[:, i, :].reshape(-1, x_hat[:, i, :].size(-1)), x[:, i].reshape(-1).long(), reduction='none')
            loss = torch.mean(w * loss_acc)
        else:
            loss = 0
            for i in range(x.size(1)):
                loss += nn.functional.cross_entropy(x_hat[:, i, :].reshape(-1, x_hat[:, i, :].size(-1)), x[:, i].reshape(-1).long(), reduction='mean')

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
