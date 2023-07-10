import json
import os
import random
import string
import uuid

from typing import Optional, Union
from pprint import pprint

import configargparse

import design_bench

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from src.diffusers import DDPMPipeline, DDPMScheduler, get_cosine_schedule_with_warmup
from util import TASKNAME2TASK, configure_gpu, set_seed, get_weights

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "diffusers"


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


class DiffusersTest(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=1024,
            learning_rate=1e-3,
            dropout_p=0,
            activation_fn=Swish(),
            num_training_steps=1000,
    ):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = self.task.y.shape[-1]
        self.dim_x = self.task.x.shape[-1]
        self.dropout_p = dropout_p
        self.num_training_steps = num_training_steps

        self.learning_rate = learning_rate

        self.model = MLP(input_dim=self.dim_x,
                         index_dim=1,
                         hidden_dim=hidden_size,
                         act=activation_fn)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_training_steps,
            tensor_format='pt',
            clip_sample=False)

    def configure_optimizers(self):
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(params=self.model.parameters(),
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
        x, y = batch
        batch_size = x.size(0)
        t_ = torch.randint(0,
                           self.noise_scheduler.num_train_timesteps, [
                               x.size(0),
                           ] + [1 for _ in range(x.ndim - 1)],
                           device=x.device)
        noise = torch.randn(x.size(), device=x.device)
        noisy_ip = self.noise_scheduler.add_noise(x, noise, t_)

        if self.dropout_p == 0:
            noise_pred = self.model(noisy_ip, t_, y)
            loss = F.mse_loss(noise_pred, noise)
        else:
            raise NotImplementedError
            # rand_mask = torch.rand(y.size())
            # mask = (rand_mask <= self.dropout_p)

            # mask randomly chosen y values
            # y[mask] = 0.
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss


class RvSDataset(Dataset):

    def __init__(self, task, x, y, w=None, device=None, mode='train'):
        self.task = task
        self.device = device
        self.mode = mode
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        if self.w is not None:
            w = torch.tensor(self.w[idx])
        else:
            w = None
        '''
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
            if w is not None:
                w = w.to(self.device)
        '''
        return x, y
        # return x, y, w


def split_dataset(task, val_frac=None, device=None):
    length = task.y.shape[0]
    shuffle_idx = np.arange(length)
    shuffle_idx = np.random.shuffle(shuffle_idx)

    x = task.x[shuffle_idx]
    y = task.y[shuffle_idx]
    x = x.reshape(-1, task.x.shape[-1])
    y = y.reshape(-1, 1)
    w = get_weights(y, base_temp=0.03 * length)

    print(w)
    print(w.shape)

    if val_frac is None:
        val_frac = 0

    val_length = int(length * val_frac)
    train_length = length - val_length

    # train_dataset = RvSDataset(task, x[:train_length], y[:train_length], w[:train_length], device, mode='train')
    # val_dataset = RvSDataset(task, x[train_length:], y[train_length:], w[train_length:], device, mode='val')
    train_dataset = RvSDataset(
        task,
        x[:train_length],
        y[:train_length],
        None,
        # w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        None,
        # w[train_length:],
        device,
        mode='val')

    return train_dataset, val_dataset


class RvSDataModule(pl.LightningDataModule):

    def __init__(self, task, batch_size, num_workers, val_frac, device):
        super().__init__()

        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.device = device
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = split_dataset(
            self.task, self.val_frac, self.device)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size)
        return val_loader


def log_args(
    args: configargparse.Namespace,
    wandb_logger: pl.loggers.wandb.WandbLogger,
) -> None:
    """Log arguments to a file in the wandb directory."""
    wandb_logger.log_hyperparams(args)

    args.wandb_entity = wandb_logger.experiment.entity
    args.wandb_project = wandb_logger.experiment.project
    args.wandb_run_id = wandb_logger.experiment.id
    args.wandb_path = wandb_logger.experiment.path

    out_directory = wandb_logger.experiment.dir
    pprint(f"out_directory: {out_directory}")
    args_file = os.path.join(out_directory, args_filename)
    with open(args_file, "w") as f:
        try:
            json.dump(args.__dict__, f)
        except AttributeError:
            json.dump(args, f)


def run_training(
    taskname: str,
    seed: int,
    wandb_logger: pl.loggers.wandb.WandbLogger,
    epochs: int,
    max_steps: int,
    train_time: str,
    hidden_size: int,
    depth: int,
    learning_rate: float,
    auto_tune_lr: bool,
    dropout_p: float,
    checkpoint_every_n_epochs: int,
    checkpoint_every_n_steps: int,
    checkpoint_time_interval: str,
    batch_size: int,
    val_frac: float,
    use_gpu: bool,
    device=None,
    num_workers=1,
    vtype='rademacher',
    num_training_steps=1,
    normalise_x=False,
    normalise_y=False,
):
    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])
    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()

    model = DiffusersTest(taskname=taskname,
                          task=task,
                          learning_rate=learning_rate,
                          hidden_size=hidden_size,
                          num_training_steps=num_training_steps,
                          dropout_p=dropout_p)

    monitor = "val_loss" if val_frac > 0 else "train_loss"
    checkpoint_dirpath = os.path.join(wandb_logger.experiment.dir,
                                      checkpoint_dir)
    checkpoint_filename = f"{taskname}_{seed}-" + "-{epoch:03d}-{" + f"{monitor}" + ":.4e}"
    periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename=checkpoint_filename,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=checkpoint_every_n_epochs,
        every_n_train_steps=checkpoint_every_n_steps,
        train_time_interval=pd.Timedelta(checkpoint_time_interval).
        to_pytimedelta() if checkpoint_time_interval is not None else None,
    )
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor=monitor,
        filename=checkpoint_filename,
        save_last=True,  # save latest model
        save_top_k=1,  # save top model based on monitored loss
    )
    trainer = pl.Trainer(
        gpus=int(use_gpu),
        auto_lr_find=auto_tune_lr,
        max_epochs=epochs,
        max_steps=max_steps,
        max_time=train_time,
        logger=wandb_logger,
        progress_bar_refresh_rate=20,
        callbacks=[periodic_checkpoint_callback, val_checkpoint_callback],
        track_grad_norm=2,  # logs the 2-norm of gradients
        limit_val_batches=1.0 if val_frac > 0 else 0,
        limit_test_batches=0,
        gradient_clip_val=1.0,
    )

    # train_dataset, val_dataset = split_dataset(task=task,
    #                                            val_frac=val_frac,
    #                                            device=device)
    # train_data_module = DataLoader(train_dataset)  #, num_workers=num_workers)
    # val_data_module = DataLoader(val_dataset)  #, num_workers=num_workers)

    # trainer.fit(model, train_data_module, val_data_module)
    data_module = RvSDataModule(task=task,
                                val_frac=val_frac,
                                device=device,
                                batch_size=batch_size,
                                num_workers=num_workers)
    trainer.fit(model, data_module)


@torch.no_grad()
def run_evaluate(
    taskname,
    seed,
    hidden_size,
    learning_rate,
    checkpoint_path,
    args,
    wandb_logger=None,
    device=None,
    normalise_x=False,
    normalise_y=False,
):
    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])
    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()

    model = DiffusersTest.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        taskname=taskname,
        task=task,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_training_steps=args.num_training_steps,
        dropout_p=args.dropout_p)

    pipeline = DDPMPipeline(unet=model.model, scheduler=model.noise_scheduler)
    bs = 256
    y = torch.ones((bs, 1)).to(device) * args.condition
    ops = pipeline(y=y, batch_size=bs, generator=torch.manual_seed(seed))
    print(ops)
    print(ops['sample'].size())
    for i in range(ops['sample'].size(0)):
        qq = ops['sample'][i, :]
        if not qq.isnan().any():
            nn = task.predict(qq[None, ...].cpu().numpy())
            print(task.denormalize_y(nn))
        else:
            print("fuck")

    yop = task.predict(ops['sample'].cpu().numpy())
    if normalise_y:
        yop = task.denormalize_y(yop)
        print(yop)
        print(yop.max())

    # save to file
    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"
    assert os.path.exists(expt_save_path)

    alias = uuid.uuid4() 
    run_specific_str = f"{bs}_{args.condition}_{args.gamma}_{args.beta_min}_{args.beta_max}_{alias}"
    save_results_dir = os.path.join(
        expt_save_path, f"wandb/latest-run/files/results/{run_specific_str}")

    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    symlink_dir = os.path.join(expt_save_path, f"wandb/latest-run/files/results/latest-run")

    print(symlink_dir)
    if os.path.exists(symlink_dir):
        os.unlink(symlink_dir)
    os.symlink(run_specific_str, symlink_dir)

    with open(os.path.join(save_results_dir, 'designs.pkl'), 'wb') as f:
        pkl.dump(ops.cpu().numpy(), f)

    with open(os.path.join(save_results_dir, 'results.pkl'), 'wb') as f:
        pkl.dump(yop, f)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        required=True)
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys()),
                        required=True)
    # reproducibility
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=
        "sets the random seed; if this is not specified, it is chosen randomly",
    )
    parser.add_argument("--condition", default=0.0, type=float)
    # experiment tracking
    parser.add_argument("--name", type=str, help="Experiment name")
    # training
    train_time_group = parser.add_mutually_exclusive_group(required=True)
    train_time_group.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="the number of training epochs.",
    )
    train_time_group.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help=
        "the number of training gradient steps per bootstrap iteration. ignored "
        "if --train_time is set",
    )
    train_time_group.add_argument(
        "--train_time",
        default=None,
        type=str,
        help="how long to train, specified as a DD:HH:MM:SS str",
    )
    parser.add_argument("--num_workers",
                        default=1,
                        type=int,
                        help="Number of workers")

    checkpoint_frequency_group = parser.add_mutually_exclusive_group(
        required=True)
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_epochs",
        default=None,
        type=int,
        help="the period of training epochs for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_steps",
        default=None,
        type=int,
        help="the period of training gradient steps for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_time_interval",
        default=None,
        type=str,
        help="how long between saving checkpoints, specified as a HH:MM:SS str",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        required=True,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="place networks and data on the GPU",
    )
    parser.add_argument("--which_gpu",
                        default=0,
                        type=int,
                        help="which GPU to use")
    parser.add_argument(
        "--normalise_x",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalise_y",
        action="store_true",
        default=False,
    )

    # i/o
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'cifar'],
                        default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--num_training_steps',
                        type=int,
                        default=1000,
                        help='integration time')
    parser.add_argument(
        '--vtype',
        type=str,
        choices=['rademacher', 'gaussian'],
        default='rademacher',
        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=1.)

    # model
    parser.add_argument(
        '--real',
        type=eval,
        choices=[True, False],
        default=True,
        help=
        'transforming the data from [0,1] to the real space using the logit function'
    )
    parser.add_argument(
        '--debias',
        type=eval,
        choices=[True, False],
        default=False,
        help=
        'using non-uniform sampling to debias the denoising score matching loss'
    )

    # TODO: remove
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="learning rate for each gradient step",
    )
    parser.add_argument(
        "--auto_tune_lr",
        action="store_true",
        default=False,
        help=
        "have PyTorch Lightning try to automatically find the best learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        help="dropout probability",
        default=0,
    )

    args = parser.parse_args()

    args.seed = np.random.randint(2**31 -
                                  1) if args.seed is None else args.seed
    set_seed(args.seed + 1)
    device = configure_gpu(args.use_gpu, args.which_gpu)

    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    if args.mode == 'train':
        if not os.path.exists(expt_save_path):
            os.makedirs(expt_save_path)
        wandb_logger = pl.loggers.wandb.WandbLogger(
            project=wandb_project,
            name=f"{args.name}_{args.seed}",
            save_dir=expt_save_path)
        log_args(args, wandb_logger)
        run_training(taskname=args.task,
                     seed=args.seed,
                     wandb_logger=wandb_logger,
                     epochs=args.epochs,
                     max_steps=args.max_steps,
                     train_time=args.train_time,
                     hidden_size=args.hidden_size,
                     depth=args.depth,
                     learning_rate=args.learning_rate,
                     auto_tune_lr=args.auto_tune_lr,
                     dropout_p=args.dropout_p,
                     checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
                     checkpoint_every_n_steps=args.checkpoint_every_n_steps,
                     checkpoint_time_interval=args.checkpoint_time_interval,
                     batch_size=args.batch_size,
                     val_frac=args.val_frac,
                     use_gpu=args.use_gpu,
                     device=device,
                     num_workers=args.num_workers,
                     vtype=args.vtype,
                     num_training_steps=args.num_training_steps,
                     normalise_x=args.normalise_x,
                     normalise_y=args.normalise_y)
    elif args.mode == 'eval':
        checkpoint_path = os.path.join(
            expt_save_path, "wandb/latest-run/files/checkpoints/last.ckpt")
        # checkpoint_path = os.path.join(
        #     expt_save_path, "wandb/latest-run/files/checkpoints/ant_123--epoch=199-val_loss=7.8251e-01.ckpt")
        run_evaluate(taskname=args.task,
                     seed=args.seed,
                     hidden_size=args.hidden_size,
                     args=args,
                     learning_rate=args.learning_rate,
                     checkpoint_path=checkpoint_path,
                     device=device,
                     normalise_x=args.normalise_x,
                     normalise_y=args.normalise_y)
    else:
        raise NotImplementedError
