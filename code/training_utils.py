""" shared utilities for training source and target models """

import os
import warnings
from argparse import ArgumentParser
from functools import partial
from os.path import join, isdir, basename
from typing import Iterator, Dict, Union

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BasePredictionWriter, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from scipy.stats import pearsonr, spearmanr
from torch import optim, Tensor
from torch.optim.lr_scheduler import LambdaLR

from . import utils
from . import datamodules
from .metrics import compute_metrics


def save_scatterplots(dm, predictions_d, log_dir, suffix=""):

    # unable to save scatterplots for sets if predictions_d only contains predictions for full dataset
    if len(predictions_d) == 1 and "full" in predictions_d:
        warnings.warn("Unable to save scatterplots for sets because predictions_d "
                      "only contains predictions for full dataset")
        return

    # save a scatter plot
    def _ss_helper(sn, preds):
        targets = dm.get_targets(sn, squeeze=True)
        if len(targets) != len(preds):
            print("Length of {} targets {} does not equal length of predictions {}. "
                  "Are you using limit_predict_batches?".format(sn, len(targets), len(preds)))
        print("saving a scatter plot for set: {} ({} variants)".format(sn, len(targets)))
        plot_scatter(targets[:len(preds)], preds, sn, log_dir, fn_suffix=suffix)

    for set_name, predictions in predictions_d.items():
        if predictions is None:
            continue
        if set_name == "full":
            # don't save scatterplot 'full' dataset
            continue
        _ss_helper(set_name, predictions)


def plot_scatter(true_scores, predicted_scores, set_name, log_dir, fn_suffix=""):

    kwargs = {"s": 8, "marker": "o", "alpha": 0.5, "edgecolor": "black", "lw": 0.1}
    ax = sns.scatterplot(x=true_scores, y=predicted_scores, **kwargs)
    fig = ax.get_figure()

    # equalize the axes, set axis limits, draw line of equivalence
    # note: outliers might mess up the scaling
    min_score = np.min([np.min(true_scores), np.min(predicted_scores)])
    max_score = np.max([np.max(true_scores), np.max(predicted_scores)])
    if not np.isnan(min_score) and not np.isnan(max_score):
        ax.set_xlim(min_score - 0.5, max_score + 0.5)
        ax.set_ylim(min_score - 0.5, max_score + 0.5)
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    bot_lef = xlims[0] if xlims[0] < ylims[0] else ylims[0]
    top_rig = ylims[1] if ylims[1] < xlims[1] else xlims[1]
    ax.plot([bot_lef, top_rig], [bot_lef, top_rig], linewidth=2, color="red", alpha=0.5)

    ax.set(xlabel="True Score", ylabel="Predicted Score")

    try:
        r = pearsonr(true_scores, predicted_scores)[0]
        p = spearmanr(true_scores, predicted_scores)[0]
    except ValueError as e:
        print("Unable to compute metrics while plotting scatterplots, probably because NaNs are present")
        r = np.nan
        p = np.nan

    anchored_text_string = "Set: {}\nPearson: {:.3f}\nSpearman: {:.3f}".format(set_name, r, p)
    anchored_text = AnchoredText(anchored_text_string, loc="upper left", frameon=False)
    ax.add_artist(anchored_text)

    fig.tight_layout()

    out_dir = join(log_dir, "plots")
    utils.mkdir(out_dir)
    fig.savefig(join(out_dir, "{}_scatter{}.png".format(set_name, fn_suffix)), dpi=150)
    plt.close(fig)


def save_metrics_custom(dm, predictions_d, log_dir, save_fn=None, suffix=""):

    # unable to save custom metrics for sets if predictions_d only contains predictions for full dataset
    if len(predictions_d) == 1 and "full" in predictions_d:
        warnings.warn("Unable to save custom metrics for sets because predictions_d "
                      "only contains predictions for full dataset")
        return None

    evaluations = {}
    for set_name, predictions in predictions_d.items():
        if predictions is None:
            continue

        if set_name == "full":
            # don't save scatterplot 'full' dataset
            continue

        targets = dm.get_targets(set_name, squeeze=True)
        # if we are running limit batches, then the length of predictions won't match the length of targets
        # this ensures we grab the same number of targets for however many batches were actually run
        if len(targets) != len(predictions):
            print("Length of {} targets {} does not equal length of predictions {}. "
                  "Are you using limit_predict_batches?".format(set_name, len(targets), len(predictions)))
        targets = targets[:len(predictions)]
        metrics = compute_metrics(targets, predictions)
        evaluations[set_name] = metrics

    # create a pandas dataframe of the evaluation metrics to save as a tsv
    sorted_order = [dm.train_name, dm.val_name, dm.test_name]
    metrics_df = pd.DataFrame(evaluations).transpose()
    metrics_df.index.rename("set", inplace=True)
    metrics_df = metrics_df.sort_index(
        key=lambda sets: [sorted_order.index(s) if s in sorted_order else len(sorted_order) for s in sets])

    print(metrics_df)

    if save_fn is None:
        save_fn = f"metrics_custom{suffix}.txt"
    metrics_df.to_csv(join(log_dir, save_fn), sep="\t")

    return metrics_df


def save_predictions(raw_preds: Union[list[torch.Tensor], list[list[torch.Tensor]]],
                     dm: datamodules.DMSDataModule,
                     log_dir: str,
                     save_format: Union[str, tuple[str]] = ("txt", "npy"),
                     suffix: str = ""):

    if isinstance(save_format, str):
        save_format = [save_format]

    # save predictions to files
    predictions_dir = join(log_dir, "predictions")
    utils.mkdir(predictions_dir)

    def _save_predictions_helper(hrp, hsn):
        np_preds = torch.cat(hrp, dim=0).cpu().numpy().squeeze()
        if "txt" in save_format:
            np.savetxt(join(predictions_dir, f"{hsn}_predictions{suffix}.txt"), np_preds, fmt="%1.7f", delimiter=",")
        if "npy" in save_format:
            np.save(join(predictions_dir, f"{hsn}_predictions{suffix}.npy"), np_preds)
        return np_preds

    # if raw_preds is a list of lists, then we have a list of predictions for each set
    # it's also possible that raw_preds is a list of lists, but one or more of the lists is None
    # this can happen if we have a datamodule with no val or test set...
    # however, we would still have the train set in that instance (index 0)...
    # if raw_preds is a list of tensors, then we have a single list of predictions for the full dataset
    if isinstance(raw_preds[0], list):
        # note these are standard set names (train, val, test), not user set names
        set_names = ["train"]
        if dm.has_val_set:
            set_names += ["val", "test"]
        else:
            set_names.append("test")

        predictions_d = {}
        for rp, sn in zip(raw_preds, set_names):
            if rp is not None:
                predictions_d[sn] = _save_predictions_helper(rp, sn)

    elif isinstance(raw_preds[0], torch.Tensor):
        # we will save the predictions for the full dataset as "full"
        predictions_d = {"full": _save_predictions_helper(raw_preds, "full")}

        # if we are using limit_predict_batches, then we will not have predictions for the full dataset
        # this will cause problems for the next step where we want to save predictions for each set
        if len(predictions_d["full"]) != len(dm.ds):
            warnings.warn("Length of predictions for 'full' dataset ({}) does not match length of datamodule ({}). "
                          "Are you using limit_predict_batches?".format(len(predictions_d["full"]), len(dm.ds)))
        else:
            # note we want to go by the datamodule standard set names, not the user-defined
            # names that might be in dm.split_idxs. this is to match past behavior, where we
            # only save predictions for the standard "train", "val" and "test" sets, whatever the user calls them
            set_names = ["train", "val", "test"]
            for sn in set_names:
                if dm.has_set(sn):
                    predictions_d[sn] = predictions_d["full"][dm.get_split_idxs(sn)]

    else:
        raise ValueError("raw_preds must be a list of lists or a list of tensors")

    return predictions_d


class PredictionWriter(BasePredictionWriter):
    def __init__(self,
                 output_dir: str,
                 save_fn_base: str = "predictions",
                 batch_write_mode: str = "combined_csv",
                 write_interval: str = "batch_and_epoch"):
        super().__init__(write_interval)

        self.output_dir = output_dir
        self.save_fn_base = save_fn_base

        self.batch_mode = None
        self.batch_format = None
        self._init_batch_write_mode(batch_write_mode)

        utils.mkdir(self.output_dir)

    def _init_batch_write_mode(self, batch_write_mode):
        valid_modes = ["combined_csv", "separate_csv", "separate_npy"]
        if batch_write_mode not in valid_modes:
            raise ValueError("batch_write_mode must be one of: {}".format(valid_modes))

        self.batch_mode = batch_write_mode.split("_")[0]  # separate or combined
        self.batch_format = batch_write_mode.split("_")[1]  # csv or npy

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if self.batch_mode == "separate" and self.batch_format == "csv":
            np.savetxt(os.path.join(self.output_dir, f"{batch_idx}.csv"),
                       prediction.cpu().numpy(), fmt="%.7f", delimiter=",")
        elif self.batch_mode == "separate" and self.batch_format == "npy":
            np.save(os.path.join(self.output_dir, f"{batch_idx}.npy"), prediction.cpu().numpy())
        elif self.batch_mode == "combined":
            # append to a combined file
            save_fn = os.path.join(self.output_dir, f"{self.save_fn_base}.csv")
            with open(save_fn, "a") as f:
                np.savetxt(f, prediction.cpu().numpy(), fmt="%.7f", delimiter=",")
        else:
            raise ValueError("Unknown batch_mode or batch_format combination. This shouldn't happen.")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # i *think* we have to use predictions[0] because first layer of lists is for dataloaders
        predictions_numpy = torch.cat(predictions[0], dim=0).cpu().numpy()
        save_fn = os.path.join(self.output_dir, f"{self.save_fn_base}.npy")
        np.save(save_fn, predictions_numpy)


class BestMetricLogger(Callback):
    def __init__(self, metric, mode):
        self.metric = metric
        self.state = {"best": torch.tensor(np.Inf) if mode == "min" else torch.tensor(-np.Inf)}

    @property
    def state_key(self):
        return self._generate_state_key(what=self.metric)

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.metric)
        if (current is not None) and (current < self.state["best"]):
            self.state["best"] = current
            pl_module.log("{}_best".format(self.metric), self.state["best"],
                          on_epoch=True, sync_dist=True, prog_bar=True)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    # https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ConstantWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup):
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 1.0
        if self.warmup == 0:
            return lr_factor
        elif epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class WarmupCosineLR(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, max_epochs, final_lr=0, last_epoch=-1, start_epoch=0):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.final_lr = final_lr
        self.start_epoch = start_epoch

        super(WarmupCosineLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.start_epoch:
            return 1.0
        elif epoch < self.start_epoch + self.warmup_epochs:
            progress = (epoch - self.start_epoch) / (self.warmup_epochs - 1)
            return progress
        else:
            progress = (epoch - self.start_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs - self.start_epoch)
            return self.final_lr + 0.5 * (1 - self.final_lr) * (1 + np.cos(np.pi * progress))


class DualPhaseConstantWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, phase2_start_step, phase2_lr_ratio=1.0):
        self.num_warmup_steps = num_warmup_steps
        self.phase2_start_step = phase2_start_step
        self.phase2_lr_ratio = phase2_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.num_warmup_steps:
            return float(epoch) / self.num_warmup_steps
        elif epoch < self.phase2_start_step:
            return 1.0
        elif epoch < self.phase2_start_step + self.num_warmup_steps:
            return (float(epoch) - self.phase2_start_step) / self.num_warmup_steps * self.phase2_lr_ratio
        else:
            return self.phase2_lr_ratio


class DualPhaseConstantWarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 total_steps: int,
                 num_warmup_steps: int,
                 phase2_start_step: int,
                 phase2_lr_ratio: float = 1.0):

        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.phase2_start_step = phase2_start_step
        self.phase2_lr_ratio = phase2_lr_ratio

        self.phase1_total_steps = self.phase2_start_step
        self.phase2_total_steps = self.total_steps - self.phase2_start_step

        print("total_steps", self.total_steps)
        print("phase1_total_steps", self.phase1_total_steps)
        print("phase2_total_steps", self.phase2_total_steps)

        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.num_warmup_steps:
            return float(epoch) / self.num_warmup_steps
        elif epoch < self.phase2_start_step:
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.phase1_total_steps))
            return lr_factor
        elif epoch < self.phase2_start_step + self.num_warmup_steps:
            return (float(epoch) - self.phase2_start_step) / self.num_warmup_steps * self.phase2_lr_ratio
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * (float(epoch) - self.phase2_start_step) / self.phase2_total_steps))
            lr_factor *= self.phase2_lr_ratio
            return lr_factor


def constant_warmup_helper(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def constant_warmup(warmup_steps):
    return partial(constant_warmup_helper, warmup_steps)


def save_metrics_ptl(best_model_path, best_model_score, test_metrics, log_dir):
    with open(join(log_dir, "metrics.txt"), "w") as f:
        f.write("best_model_path,{}\n".format(best_model_path))
        f.write("best_model_score,{}\n".format(best_model_score))
        for k, v in test_metrics[0].items():
            f.write("{},{}\n".format(k, v))


def plot_losses(log_dir):
    metrics_fn = join(log_dir, "metrics.csv")
    metrics_df = pd.read_csv(metrics_fn)

    # create loss dataframe (with val loss if a val set was used)
    loss_df = metrics_df[metrics_df["train_loss_epoch"].notnull()][["epoch", "train_loss_epoch"]]
    if "val_loss" in metrics_df:
        val_loss_df = metrics_df[metrics_df["val_loss"].notnull()][["epoch", "val_loss"]]
        loss_df = pd.merge(loss_df, val_loss_df, on="epoch")
    loss_df = loss_df.set_index("epoch", verify_integrity=True)

    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax = sns.lineplot(data=loss_df, ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    label_dict = {"train_loss_epoch": "Train", "val_loss": "Val"}
    legend = ax.get_legend()
    if legend is not None:
        for label in legend.get_texts():
            label.set_text(label_dict.get(label.get_text(), label.get_text()))

    ax.set(ylabel="Loss", xlabel="Epoch", title="Training and validation loss")

    out_dir = join(log_dir, "plots")
    utils.mkdir(out_dir)

    fig.savefig(join(out_dir, "losses.png"), dpi=150)
    plt.close(fig)


def plot_losses_source_model(log_dir):
    metrics_fn = join(log_dir, "metrics.csv")
    metrics_df = pd.read_csv(metrics_fn)

    if "val_loss" in metrics_df.columns and "train_loss_epoch" in metrics_df.columns:
        val_loss_df = metrics_df[metrics_df["val_loss"].notnull()][["epoch", "val_loss"]]
        train_loss_df = metrics_df[metrics_df["train_loss_epoch"].notnull()][["epoch", "train_loss_epoch"]]
        loss_df = pd.merge(train_loss_df, val_loss_df, on="epoch").set_index("epoch", verify_integrity=True)
    elif "train_loss_epoch" in metrics_df.columns and "val_loss" not in metrics_df.columns:
        loss_df = metrics_df[metrics_df["train_loss_epoch"].notnull()][["epoch", "train_loss_epoch"]]
        loss_df = loss_df.set_index("epoch", verify_integrity=True)
    else:
        # no train_loss_epoch --> shouldn't happen
        # no train_loss_epoch and no val_loss --> weird edge case that came up during development
        return

    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax = sns.lineplot(data=loss_df, ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(labels=["Train", "Val"])
    ax.set(ylabel="Loss", xlabel="Epoch", title="Training and Validation Loss")

    out_dir = join(log_dir, "plots")
    utils.mkdir(out_dir)

    fig.savefig(join(out_dir, "losses.png"), dpi=150)
    plt.close(fig)


class CondorStopping(Callback):
    def __init__(self, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.stopped = False

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip_check(trainer):
            return

        self._run_condor_stopping_check(trainer)

    def _run_condor_stopping_check(self, trainer: "pl.Trainer") -> None:
        if trainer.fast_dev_run:
            # disable early_stopping if fast_dev_run via short circuit
            return

        # determine if we meet the criteria to stop training (every_n_epochs)
        should_stop = self._evaluate_stopping_criteria(trainer)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped = True

    def _evaluate_stopping_criteria(self, trainer) -> bool:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            return True
        else:
            return False


def create_log_dir(log_dir_base, given_uuid):
    if given_uuid == "condor":
        # this is a special value that instructs prep_condor_run to generate UUIDs
        # this should not actually make it to this script as an input
        raise ValueError("UUID 'condor' is a special value for prep_condor_run")

    # set up log directory & save the args file to it
    if given_uuid is None:
        # script was not given a custom UUID
        my_uuid = utils.gen_model_uuid()
        print("Created model UUID: {}".format(my_uuid))
        log_dir = utils.log_dir_name(log_dir_base, my_uuid)
        os.makedirs(log_dir, exist_ok=True)
        print("Created log directory: {}".format(log_dir))
    else:
        # script was given a custom UUID
        print("User gave model UUID: {}".format(given_uuid))
        my_uuid = given_uuid

        # check if a log directory already exists for this UUID
        existing_log_dir, log_dir = check_for_existing_log_dir(log_dir_base, my_uuid)
        if not existing_log_dir:
            # did not find an existing log directory, create our own using the supplied UUID
            print("Did not find existing log directory corresponding to given UUID: {}".format(my_uuid))
            log_dir = utils.log_dir_name(log_dir_base, my_uuid)
            os.makedirs(log_dir, exist_ok=True)
            print("Created log directory: {}".format(log_dir))

    # at this point, my_uuid and log_dir are set to correct values, regardless of whether
    # uuid was passed in or created fresh in this script
    print("Final UUID: {}".format(my_uuid))
    print("Final log directory: {}".format(log_dir))

    return my_uuid, log_dir


def check_for_existing_log_dir(log_dir_base, my_uuid):
    existing_log_dir = False

    if not isdir(log_dir_base):
        return False, None

    # looking within the log_dir_base directory
    log_dirs = [join(log_dir_base, x) for x in os.listdir(log_dir_base) if isdir(join(log_dir_base, x))]

    # simply see if any of the log directory names contain the given UUID
    log_dir = None
    for ld in log_dirs:
        if my_uuid in basename(ld).split("_"):
            print("Found existing log directory corresponding to given UUID: {}".format(ld))
            log_dir = ld
            existing_log_dir = True
            break

    return existing_log_dir, log_dir


def get_next_version(log_dir):
    # adapted from Pytorch Lightning's CSVLogger
    existing_versions = []
    for d in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def init_loggers(log_dir,
                 my_uuid,
                 use_wandb,
                 wandb_online,
                 wandb_project) -> Union[tuple[WandbLogger, TensorBoardLogger, CSVLogger],
                                         tuple[TensorBoardLogger, CSVLogger]]:
    """ set up logger callbacks for trainer """
    tb_logger = TensorBoardLogger(
        save_dir=join(log_dir, "tensorboard_{}".format(my_uuid)),
        name="",
        version="",
        log_graph=False
    )
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            save_dir=log_dir,
            id=my_uuid,
            name=my_uuid,
            offline=not wandb_online,
            project=wandb_project,
            settings=wandb.Settings(symlink=False)
        )
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="",
        version=""
    )

    if use_wandb:
        return wandb_logger, tb_logger, csv_logger
    else:
        return tb_logger, csv_logger


class OptimizerConfig:
    """ reusable optimizer configuration """

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        p.add_argument('--optimizer', type=str, default="adamw", choices=["sgd", "adamw", "adam"])
        p.add_argument('--weight_decay', type=float, default=0.01,
                       help="weight decay parameter for optimizer")
        p.add_argument('--learning_rate', type=float, default=0.0001)
        p.add_argument('--lr_scheduler', type=str, default="constant",
                       choices=["constant", "warmup_constant", "warmup_cosine_decay",
                                "dual_phase_warmup_constant", "dual_phase_warmup_constant_cosine_decay"]),
        p.add_argument('--warmup_steps', type=float, default=.02,
                       help="number or fraction of warmup steps for warmup_cosine_decay")
        p.add_argument('--phase2_lr_ratio', type=float, default=1.0,
                       help="phase 2 lr ratio for dual_phase_warmup_constant lr scheduler")
        return p

    def __init__(self,
                 optimizer: str,
                 weight_decay: float,
                 learning_rate: float,
                 lr_scheduler: str,
                 # for lr schedulers that use warmup
                 warmup_steps: float,
                 # for dual_phase lr schedulers
                 phase2_lr_ratio: float,
                 unfreeze_backbone_at_epoch: int = None,
                 max_epochs: int = None,
                 *args, **kwargs):

        super().__init__()

        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.phase2_lr_ratio = phase2_lr_ratio

        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.max_epochs = max_epochs

        self.error_checking()

    def error_checking(self):
        if self.optimizer == "adam" and self.weight_decay != 0:
            warnings.warn("Optimizer is set to adam (not adamW) with weight_decay={}. "
                          "If using weight_decay!=0, recommend using adamW ".format(self.weight_decay))

    def init_optimizer(self, trainable_parameters):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("unsupported value for optimizer: {}".format(self.optimizer))
        return optimizer

    def calc_warmup_steps(self, estimated_stepping_batches):
        # calculate the number of warmup steps (could be absolute number or fraction of total steps)
        ws = self.warmup_steps
        if 0 < self.warmup_steps < 1:
            ws = self.warmup_steps * estimated_stepping_batches
        return ws

    def get_optimizer_config(self,
                             trainable_parameters: Iterator[torch.nn.Parameter],
                             estimated_stepping_batches: int):

        optimizer = self.init_optimizer(trainable_parameters)

        if self.lr_scheduler == "warmup_cosine_decay":
            ws = self.calc_warmup_steps(estimated_stepping_batches)
            print("Number of training steps is {}".format(estimated_stepping_batches))
            print("Number of warmup steps is {}".format(ws))

            lr_scheduler_config = {
                "scheduler": CosineWarmupScheduler(optimizer, warmup=ws, max_iters=estimated_stepping_batches),
                "interval": "step",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        elif self.lr_scheduler == "warmup_constant":
            ws = self.calc_warmup_steps(estimated_stepping_batches)
            print("Number of training steps is {}".format(estimated_stepping_batches))
            print("Number of warmup steps is {}".format(ws))

            # set up the learning rate scheduler configuration
            lr_scheduler_config = {
                "scheduler": ConstantWarmupScheduler(optimizer, warmup=ws),
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        elif self.lr_scheduler in ["dual_phase_warmup_constant", "dual_phase_warmup_constant_cosine_decay"]:
            # a special warmup scheduler configured to do another round of warmup
            # halfway through training... meant for use w/ finetuning callback
            ws = self.calc_warmup_steps(estimated_stepping_batches)
            print("Number of training steps is {}".format(estimated_stepping_batches))
            print("Number of warmup steps is {}".format(ws))

            # if both unfreeze_backbone_at_epoch and max_epochs are set, then
            # we can calculate the step at which to start the second phase of warmup
            # otherwise, we just use the default of halfway through training
            if self.unfreeze_backbone_at_epoch is not None and self.max_epochs is not None:
                # determine the step at which to start the second phase of warmup
                # based on the estimated number of steps per epoch and the epoch at which to unfreeze the backbone
                steps_per_epoch = estimated_stepping_batches / self.max_epochs
                phase2_start_step = int(self.unfreeze_backbone_at_epoch * steps_per_epoch)
            else:
                # default to halfway through training
                phase2_start_step = int(estimated_stepping_batches / 2)

            print("Second warmup phase starts at step {}".format(phase2_start_step))

            # set up the learning rate scheduler configuration
            if self.lr_scheduler == "dual_phase_warmup_constant":
                scheduler = DualPhaseConstantWarmupScheduler(optimizer=optimizer,
                                                             num_warmup_steps=ws,
                                                             phase2_start_step=phase2_start_step,
                                                             phase2_lr_ratio=self.phase2_lr_ratio)
            elif self.lr_scheduler == "dual_phase_warmup_constant_cosine_decay":
                scheduler = DualPhaseConstantWarmupCosineDecayScheduler(optimizer=optimizer,
                                                                        total_steps=estimated_stepping_batches,
                                                                        num_warmup_steps=ws,
                                                                        phase2_start_step=phase2_start_step,
                                                                        phase2_lr_ratio=self.phase2_lr_ratio)
            else:
                # this shouldn't happen
                raise ValueError("unknown learning rate scheduler: {}".format(self.lr_scheduler))

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        elif self.lr_scheduler == "constant":
            # temporary workaround for finetuning scheduler, which requires a scheduler config
            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0),
                "interval": "step",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            raise ValueError("unknown learning rate scheduler: {}".format(self.lr_scheduler))


class DelayedStartEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set start_epoch to None or 0 for no delay
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super().on_validation_end(trainer, pl_module)


class DelayedStartModelCheckpoint(ModelCheckpoint):
    """ only starts saving topk/monitored checkpoints after start_epoch """
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def _save_topk_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        # do not save or update related state for topk checkpoints until the start epoch
        if (self.start_epoch is not None) and (trainer.current_epoch < self.start_epoch):
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class SimpleProgressMessages(Callback):

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Starting sanity check...")

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Sanity check complete.")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Starting training...")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            # don't print progress during the sanity check
            return
        train_loss_epoch = trainer.callback_metrics.get('train_loss_epoch', np.nan)
        val_loss = trainer.callback_metrics.get('val_loss', np.nan)
        print(f"Epoch {trainer.current_epoch:>5}: Train Loss = {train_loss_epoch:>7.3f}, Val Loss = {val_loss:>7.3f}")

    def on_test_start(self, trainer, pl_module):
        print("Starting testing...")

    def on_test_epoch_end(self, trainer, pl_module):
        print("Testing complete.")

    def on_predict_start(self, trainer, pl_module):
        print("Starting prediction...")

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        print("Prediction complete.")
