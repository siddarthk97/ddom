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


import numpy as np
import pytorch_lightning as pl

import torch
from torch import optim, nn, utils, Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from util import TASKNAME2TASK

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            hidden_dim=128,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.act = act
        self.y_dim = 1
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.y_dim),
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

    def forward(self, input):
        # init
        input = input.view(-1, self.input_dim)
        # t = t.view(-1, self.index_dim).float()
        # y = y.view(-1, self.y_dim).float()

        # forward
        # h = torch.cat([input, t, y], dim=1)  # concat
        h = input
        output = self.main(h)  # forward
        return output

class ForwardModel(pl.LightningModule):

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
        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate

        self.mlp = MLP(input_dim=self.dim_x,
                           hidden_dim=hidden_size,
                           act=activation_fn)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = torch.optim.Adam(self.mlp.parameters(),
                                     lr=self.learning_rate)
        return optimizer

    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.mlp.parameters(),
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
        pred = self.mlp(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

