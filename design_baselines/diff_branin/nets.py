import sys
import os
import math

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
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from util import TASKNAME2TASK

from lib.sdes import VariancePreservingSDE, PluginReverseSDE, ScorePluginReverseSDE
from unet import UNET_1D


def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            index_dim=1,
            hidden_dim=128,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act
        self.y_dim = 1
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, input_dim),
        )
        """
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        """

    def forward(self, input, t, y):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()
        y = y.view(-1, self.y_dim).float()

        # forward
        h = torch.cat([input, t, y], dim=1)  # concat
        output = self.main(h)  # forward
        return output.view(*sz)


class DiffusionTest(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=1024,
            learning_rate=1e-3,
            # activation_fn=nn.ReLU(),
            beta_min=0.1,
            beta_max=20.0,
            dropout_p=0,
            simple_clip=False,
            activation_fn=Swish(),
            T0=1,
            debias=False,
            vtype='rademacher'):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = self.task.y.shape[-1]
        if not task.is_discrete:
            self.dim_x = self.task.x.shape[-1]
        else:
            self.dim_x = self.task.x.shape[-1] * self.task.x.shape[-2]
        self.dropout_p = dropout_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.simple_clip = simple_clip
        self.debias = debias

        self.clip_min = torch.tensor(task.x).min(axis=0)[0]
        self.clip_max = torch.tensor(task.x).max(axis=0)[0]

        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate

        self.drift_q = MLP(input_dim=self.dim_x,
                           index_dim=1,
                           hidden_dim=hidden_size,
                           act=activation_fn)
        # self.drift_q = UNET_1D(1, 128, 7, 3)
        self.T = torch.nn.Parameter(torch.FloatTensor([self.T0]),
                                    requires_grad=False)

        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)
        self.gen_sde = PluginReverseSDE(self.inf_sde,
                                        self.drift_q,
                                        self.T,
                                        vtype=self.vtype,
                                        debias=self.debias)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = torch.optim.Adam(self.gen_sde.parameters(),
                                     lr=self.learning_rate)
        return optimizer

    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.gen_sde.parameters(),
                                     lr=self.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            # TODO: add to config
            num_warmup_steps=500,
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(10004 * 1000),
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "epoch"}
        return [optimizer], [scheduler]
    """

    def training_step(self, batch, batch_idx, log_prefix="train"):
        # x, y = batch
        x, y, w = batch
        self.clip_min.cuda()
        self.clip_max.cuda()
        if self.dropout_p == 0:
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x, y, w,
                clip=self.simple_clip).mean()  # forward and compute loss
        else:
            rand_mask = torch.rand(y.size())
            mask = (rand_mask <= self.dropout_p)

            # mask randomly chosen y values
            y[mask] = 0.
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x,
                y,
                w,
                clip=self.simple_clip,
                c_min=self.clip_min,
                c_max=self.clip_max).mean()  # forward and compute loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.training_step(batch, batch_idx, log_prefix="val")
        x, y, w = batch
        loss = self.gen_sde.elbo_random_t_slice(x, y)
        self.log(f"elbo_estimator", loss, prog_bar=True)
        return loss


class DiffusionScore(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=1024,
            learning_rate=1e-3,
            # activation_fn=nn.ReLU(),
            beta_min=0.1,
            beta_max=20.0,
            dropout_p=0,
            simple_clip=False,
            activation_fn=Swish(),
            T0=1,
            debias=False,
            vtype='rademacher'):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = self.task.y.shape[-1]
        self.dim_x = self.task.x.shape[-1]
        self.dropout_p = dropout_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.simple_clip = simple_clip
        self.debias = debias

        self.clip_min = torch.tensor(task.x).min(axis=0)[0]
        self.clip_max = torch.tensor(task.x).max(axis=0)[0]

        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate

        self.score_estimator = MLP(input_dim=self.dim_x,
                                   index_dim=1,
                                   hidden_dim=hidden_size,
                                   act=activation_fn)
        self.T = torch.nn.Parameter(torch.FloatTensor([self.T0]),
                                    requires_grad=False)

        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)
        self.gen_sde = ScorePluginReverseSDE(self.inf_sde,
                                             self.score_estimator,
                                             self.T,
                                             vtype=self.vtype,
                                             debias=self.debias)

    # def configure_optimizers(self) -> optim.Optimizer:
    #     """Configures the optimizer used by PyTorch Lightning."""
    #     optimizer = torch.optim.Adam(self.gen_sde.parameters(), lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.gen_sde.parameters(),
                                     lr=self.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            # TODO: add to config
            num_warmup_steps=500,
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(10004 * 1000),
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "epoch"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, log_prefix="train"):
        # x, y = batch
        x, y, w = batch
        self.clip_min.cuda()
        self.clip_max.cuda()
        if self.dropout_p == 0:
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x, y, w,
                clip=self.simple_clip).mean()  # forward and compute loss
        else:
            rand_mask = torch.rand(y.size())
            mask = (rand_mask <= self.dropout_p)

            # mask randomly chosen y values
            y[mask] = 0.
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x,
                y,
                w,
                clip=self.simple_clip,
                c_min=self.clip_min,
                c_max=self.clip_max).mean()  # forward and compute loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.training_step(batch, batch_idx, log_prefix="val")
        x, y, w = batch
        loss = self.gen_sde.elbo_random_t_slice(x, y)
        self.log(f"elbo_estimator", loss, prog_bar=True)
        return loss
