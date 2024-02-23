""" training source model to predict rosetta energies """
import sys
import warnings
from os.path import join, basename, isfile
import os
from argparse import ArgumentParser
from typing import Union

import torch
import torch.distributed
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import pytorch_lightning.accelerators

import wandb

import utils
from training_utils import BestMetricLogger, save_metrics_ptl, CondorStopping, create_log_dir, get_next_version
from datamodules import RosettaDataModule
import models
import tasks


class ModelCheckpoint(pytorch_lightning.callbacks.ModelCheckpoint):
    """ this is a custom ModelCheckpoint that saves the best model paths as relative paths.
        this was necessary for loading checkpoints on HTCondor  where the abs path to the log directory changes.
        later versions of PyTorch Lightning might do this by default, so this class may not be necessary anymore.
        will need to check when updating to a newer version of PyTorch Lightning. """

    def __init__(self, dirpath, *args, **kwargs):
        super(ModelCheckpoint, self).__init__(dirpath, *args, **kwargs)

        # the default init function calls dirpath = os.path.realpath(dirpath)
        # we're just going to use the relative path...
        self.dirpath = os.path.relpath(dirpath)

    def to_rel(self, abs_path):
        if abs_path == "":
            return abs_path
        else:
            return join(self.dirpath, basename(abs_path))

    # for compatability with old checkpoints that used an absolute dirpath, need to modify
    # the checkpoint's state_dict to change it to relative when it gets loaded from disk
    def load_state_dict(self, state_dict):
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)
        if not os.path.isabs(dirpath_from_ckpt):
            # already have a relative path from checkpoint, just run standard load_state_dict
            super(ModelCheckpoint, self).load_state_dict(state_dict)
        else:
            # we have an absolute path from checkpoint, convert state to relative paths if suffixes match
            # essentially checking if the absolute path in the checkpoint ends with
            # the relative path from the new ModelCheckpoint object... if so, assume same dirpath
            suffix = os.path.commonprefix([state_dict["dirpath"][::-1], self.dirpath[::-1]])[::-1]
            if suffix != self.dirpath:
                warnings.warn(
                    f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                    " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                    " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
                )
            else:
                state_dict["dirpath"] = self.dirpath
                self.best_model_score = state_dict["best_model_score"]
                self.kth_best_model_path = self.to_rel(state_dict.get("kth_best_model_path", self.kth_best_model_path))
                self.kth_value = state_dict.get("kth_value", self.kth_value)
                self.best_k_models = {self.to_rel(k): v for k, v in
                                      state_dict.get("best_k_models", self.best_k_models).items()}
                self.last_model_path = self.to_rel(state_dict.get("last_model_path", self.last_model_path))
                self.best_model_path = self.to_rel(state_dict["best_model_path"])


def create_log_dir_version(log_dir):
    # figure out version number for this run (in case we are resuming a check-pointed run)
    version = get_next_version(log_dir)
    print("This is version: {}".format(version))

    # the log directory for this version
    log_dir_version = join(log_dir, "version_{}".format(version))
    os.makedirs(log_dir_version, exist_ok=True)
    print("Version-specific logs will be saved to: {}".format(log_dir_version))

    return version, log_dir_version


def get_checkpoint_path(log_dir):
    if isfile(join(log_dir, "checkpoints", "last.ckpt")):
        ckpt_path = join(log_dir, "checkpoints", "last.ckpt")
        print("Found checkpoint, resuming training from: {}".format(ckpt_path))
    else:
        ckpt_path = None
        print("No checkpoint found, training from scratch")
    return ckpt_path


def init_loggers(log_dir: str,
                 my_uuid: str,
                 use_wandb: bool,
                 wandb_online: bool,
                 wandb_project: str,
                 version: int) -> Union[tuple[WandbLogger, CSVLogger], CSVLogger]:

    # set up loggers for training
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="",
        version=version
    )

    if use_wandb:
        wandb_logger = WandbLogger(
            save_dir=log_dir,
            id=my_uuid,
            name=my_uuid,
            offline=not wandb_online,
            project=wandb_project,
            settings=wandb.Settings(symlink=False)
        )
        loggers = (wandb_logger, csv_logger)
        return loggers

    else:
        return csv_logger


def get_encoding(args):
    if args.encoding == "auto":
        # for backwards compatibility with old approach of choosing encoding based on model type
        # this can be removed in future versions
        if args.model_name in ["linear", "fully_connected", "cnn"]:
            encoding = "one_hot"
        elif args.model_name in ["transformer_encoder", "cnn2"]:
            encoding = "int_seqs"
        else:
            raise ValueError("unknown encoding for model name: {}".format(args.model_name))
    else:
        encoding = args.encoding
    return encoding


def main(args):

    # random seed
    pl.seed_everything(args.random_seed)

    # GPU and distributed training config
    # auto accelerator will automatically use GPU if available, CPU otherwise
    # auto devices will automatically use the max available GPU and CPU
    # only thing I'm not sure about is strategy, so I explicitly set it to DDP if using more than one GPU
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    devices = "auto"
    strategy = None
    if torch.cuda.device_count() > 1:
        print("Detected {} CUDA devices, setting strategy to DDP".format(torch.cuda.device_count()))
        strategy = "ddp"

    # set up log directory & save the args file to it, only on rank 0
    if os.getenv("LOCAL_RANK", '0') == '0':
        # get the uuid and log directory for this run
        my_uuid, log_dir = create_log_dir(args.log_dir_base, args.uuid)

        # get the version and version log directory for this run
        # the version log directory is contained within the main log directory
        # the version number starts at 0 on the first run for this UUID
        # a new version is created every time this model UUID run is restarted
        version, log_dir_version = create_log_dir_version(log_dir)

        # set environment variables just in case we are running in DDP
        # the other ranks will get these values from the environment instead of
        os.environ["PL_LOG_DIR"] = log_dir
        os.environ["PL_UUID"] = my_uuid
        os.environ["PL_VERSION"] = str(version)
        os.environ["PL_LOG_DIR_VERSION"] = log_dir_version

        utils.save_args(vars(args), join(log_dir_version, "args.txt"), ignore=["cluster", "process"])

    elif "PL_LOG_DIR" in os.environ and "PL_UUID" in os.environ \
            and "PL_VERSION" in os.environ and "PL_LOG_DIR_VERSION" in os.environ:
        # executing on non-rank 0, get the log directory and uuid from environment variables
        log_dir = os.environ["PL_LOG_DIR"]
        my_uuid = os.environ["PL_UUID"]
        version = int(os.environ["PL_VERSION"])
        log_dir_version = os.environ["PL_LOG_DIR_VERSION"]
    else:
        # executing on non-rank 0, but expected environment variables are not set
        raise ValueError("PL_LOG_DIR or PL_UUID or PL_VERSION or PL_LOG_DIR_VERSION"
                         " environment variables not set on rank {}".format(os.getenv("LOCAL_RANK", '0')))

    # are we resuming from checkpoint, and if so, what is the checkpoint path
    # assumes the latest checkpoint is called last.ckpt saved in the checkpoints directory
    ckpt_path = get_checkpoint_path(log_dir)

    # set up loggers for training
    loggers = init_loggers(log_dir, my_uuid, args.use_wandb, args.wandb_online, args.wandb_project, version)

    # set up the datamodule
    encoding = get_encoding(args)
    dm = RosettaDataModule(ds_fn=args.ds_fn,
                           encoding=encoding,
                           target_group=args.target_group,
                           target_names=args.target_names,
                           target_names_exclude=args.target_names_exclude,
                           split_dir=args.split_dir,
                           train_name=args.train_name,
                           val_name=args.val_name,
                           test_name=args.test_name,
                           batch_size=args.batch_size,
                           enable_distributed_sampler=True if strategy == "ddp" else False,
                           enable_pdb_sampler=True,
                           use_padding_collate_fn=False)

    # set up the RosettaTask
    # pass in arguments from the datamodule that are important for model construction
    # other important args, like model_name, learning_rate, etc., are in the argparse args object
    task = tasks.RosettaTask(num_tasks=dm.num_tasks,
                             num_tokens=dm.num_tokens,
                             aa_seq_len=dm.aa_seq_len,
                             aa_encoding_len=dm.aa_encoding_len,
                             seq_encoding_len=dm.seq_encoding_len,
                             pdb_fns=dm.unique_pdb_fns,
                             example_input_array=dm.example_input_array,
                             **vars(args))

    # set up callbacks
    callbacks = [BestMetricLogger(metric="val_loss", mode="min"),
                 RichModelSummary(max_depth=-1),
                 LearningRateMonitor()]

    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=args.es_min_delta,
            patience=args.es_patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    if args.condor_checkpoint_every_n_epochs > 0:
        condor_stopping_callback = CondorStopping(every_n_epochs=args.condor_checkpoint_every_n_epochs)
        callbacks.append(condor_stopping_callback)

    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(log_dir, "checkpoints"),
        filename="{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=5,
        mode="min",
        auto_insert_metric_name=True,
        every_n_epochs=1
    )
    callbacks.append(checkpoint_callback)

    # checkpoints at regular intervals (every 10 epochs)
    checkpoint_callback_2 = ModelCheckpoint(
        dirpath=join(log_dir, "checkpoints", "interval_checkpoints"),
        every_n_epochs=10,
        save_top_k=-1
    )
    callbacks.append(checkpoint_callback_2)

    # set trainer strategy to find_unused_params false in case of DDP
    trainer_strategy = strategy
    if trainer_strategy == "ddp":
        trainer_strategy = DDPStrategy(find_unused_parameters=False)

    # set up wandb to log gradients, parameter histograms
    if args.use_wandb and args.wandb_log_grads:
        loggers[0].watch(task, log="all", log_freq=args.grad_log_freq)

    # set up the trainer from argparse args
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=log_dir,
                                            callbacks=callbacks,
                                            logger=loggers,
                                            accelerator=accelerator, devices=devices, strategy=trainer_strategy,
                                            replace_sampler_ddp=False)

    # run training
    trainer.fit(task, datamodule=dm, ckpt_path=ckpt_path)

    # destroy the DDP process group after training -- everything from here should just run on 1 GPU
    if strategy == "ddp":
        torch.distributed.destroy_process_group()

    # did we stop due to condor checkpoint stopping?
    if (args.condor_checkpoint_every_n_epochs > 0) and condor_stopping_callback.stopped:
        # I believe this only happens on rank 0 because the process group is destroyed above after fit
        # exit with code 85
        sys.exit(85)

    # print out best checkpoint paths
    print(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_score)

    # run test set
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/8375
    if strategy == "ddp":
        # torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = pl.Trainer(accelerator=accelerator,
                                 devices=1,
                                 strategy=None,
                                 logger=loggers,
                                 default_root_dir=log_dir)
            model = tasks.RosettaTask.load_from_checkpoint(checkpoint_callback.best_model_path)
            test_metrics = trainer.test(model, datamodule=dm)

    else:
        test_metrics = trainer.test(ckpt_path="best", datamodule=dm)

    if trainer.is_global_zero:
        # save metrics and losses (*at the best epoch*) to csv
        save_metrics_ptl(checkpoint_callback.best_model_path,
                         checkpoint_callback.best_model_score, test_metrics, log_dir_version)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)

    # Program args
    parser.add_argument("--random_seed",
                        help="random seed",
                        type=int,
                        default=1)

    # early stopping
    parser.add_argument("--early_stopping",
                        help="set this flag to enable early stopping",
                        action="store_true")
    parser.add_argument("--es_patience",
                        help="number of epochs allowance for early stopping",
                        type=int,
                        default=5)
    parser.add_argument("--es_min_delta",
                        help="the min amount by which the loss must decrease. if the loss does not decrease by this "
                             "amount for the given allowance of epochs, then training is considered complete",
                        type=float,
                        default=0.001)

    # HTCondor args
    parser.add_argument("--cluster",
                        help="cluster (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--process",
                        help="process (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--github_tag",
                        help="github tag for current run",
                        type=str,
                        default="no_github_tag")

    # additional args
    parser.add_argument("--log_dir_base",
                        help="log directory base",
                        type=str,
                        default="output/training_logs")
    parser.add_argument("--uuid",
                        help="model uuid to resume from or custom uuid to use from scratch",
                        type=str,
                        default=None)
    parser.add_argument("--condor_checkpoint_every_n_epochs",
                        help="how often to perform a condor checkpoint (exit with code 85)",
                        type=int,
                        default=0)

    # wandb args
    parser.add_argument('--use_wandb', action='store_true',
                        help="use wandb for logging")
    parser.add_argument('--no_use_wandb', dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb_online",
                        action="store_true",
                        default=False)
    parser.add_argument("--wandb_project",
                        type=str,
                        default="metl")
    parser.add_argument("--wandb_log_grads",
                        default=False,
                        action="store_true",
                        help="whether to log gradients and parameter histograms to weights&biases")
    parser.add_argument("--grad_log_freq",
                        default=500,
                        type=int,
                        help="log frequency for gradients")
    parser.add_argument("--experiment",
                        type=str,
                        default="default",
                        help="dummy arg to make wandb tracking and filtering easier")

    # add data specific args
    parser = RosettaDataModule.add_data_specific_args(parser)

    # add task-specific specific args
    parser = tasks.RosettaTask.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    # need to have this additional argument parser line to add the fromfile_prefix_chars
    # this lets us specify the model_name in the file along with model specific args
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)
    parser.add_argument("--model_name", type=str, choices=[m.name for m in list(models.Model)])

    # this line is key to pull the model name
    temp_args, _ = parser.parse_known_args()

    # add the model-specific arguments for the given model_name, if add_model_specific_args() exists
    add_args_op = getattr(models.Model[temp_args.model_name].cls, "add_model_specific_args", None)
    if callable(add_args_op):
        parser = add_args_op(parser)

    # finally, make sure we can use args from file (can't do this before because it gets overwritten)
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)

    parsed_args = parser.parse_args()

    main(parsed_args)
