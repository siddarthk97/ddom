import json
import os

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

from nets import RvSContinuous, RvSDiscrete
from util import TASKNAME2TASK, configure_gpu, set_seed, get_weights

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "rvs"


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
        # return x, y
        return x, y, w


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
        # None,
        w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        # None,
        w[train_length:],
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
    normalise=False,
):
    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])
    if normalise:
        # task.map_normalize_x()
        task.map_normalize_y()

    if task.is_discrete:
        model_type = RvSDiscrete
    else:
        model_type = RvSContinuous

    model = model_type(taskname=taskname,
                       task=task,
                       hidden_size=hidden_size,
                       learning_rate=learning_rate)

    wandb_logger.watch(model, log="all")

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
def run_evaluate(taskname,
                 seed,
                 hidden_size,
                 learning_rate,
                 checkpoint_path,
                 args,
                 wandb_logger=None,
                 device=None,
                 normalise=False):
    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])
    if normalise:
        # task.map_normalize_x()
        task.map_normalize_y()

    if task.is_discrete:
        model_type = RvSDiscrete
    else:
        model_type = RvSContinuous

    model = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                            taskname=taskname,
                                            task=task)
    model = model.to(device)

    model.eval()

    # ip = torch.arange(256)
    # ip = torch.arange(1024)
    if normalise:
        ip = torch.arange(-3, 3, 0.01)
    else:
        ip = torch.arange(512)
    # ip = ip+1
    # ip = ip + 100

    ip = ip.view(-1, 1)
    ip = ip.float()

    ip = ip.to(device)
    op = model(ip)

    print(op.size())

    if task.is_discrete:
        op = torch.argmax(op, axis=1)
        op = op.unsqueeze(dim=-1)

    gtop = task.predict(op.cpu().numpy())
    if normalise:
        gtop = task.denormalize_y(gtop)
    print(gtop)
    print(gtop.shape)

    # save to file
    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"
    assert os.path.exists(expt_save_path)

    save_results_dir = os.path.join(expt_save_path,
                                    "wandb/latest-run/files/results/")
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    with open(os.path.join(save_results_dir, 'designs.pkl'), 'wb') as f:
        pkl.dump(op.cpu().numpy(), f)

    with open(os.path.join(save_results_dir, 'results.pkl'), 'wb') as f:
        pkl.dump(gtop, f)


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
    # architecture
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
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
        required=True,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=True,
        help="dropout probability",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size for each gradient step",
    )
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
    parser.add_argument(
        "--normalise",
        action="store_true",
        default=False,
    )
    parser.add_argument("--which_gpu",
                        default=0,
                        type=int,
                        help="which GPU to use")

    args = parser.parse_args()

    args.seed = np.random.randint(2**31 - 1) if args.seed is None else args.seed
    set_seed(args.seed + 1)
    device = configure_gpu(args.use_gpu, args.which_gpu)

    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    print(args.normalise)
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
                     normalise=args.normalise)
    elif args.mode == 'eval':
        checkpoint_path = os.path.join(
            expt_save_path, "wandb/latest-run/files/checkpoints/last.ckpt")
        run_evaluate(taskname=args.task,
                     seed=args.seed,
                     hidden_size=args.hidden_size,
                     args=args,
                     learning_rate=args.learning_rate,
                     checkpoint_path=checkpoint_path,
                     device=device)
    else:
        raise NotImplementedError
