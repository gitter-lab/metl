""" Run inference with METL models."""

import argparse
import warnings
from os.path import join, isfile, basename, dirname
from typing import Optional, Any, Type

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelSummary
from torch import nn

try:
    # for running locally on Apple Silicon (only available in PyTorch 1.12+)
    import torch.backends.mps
    mps_available = torch.backends.mps.is_available()
except ModuleNotFoundError:
    mps_available = False

import datamodules
import tasks
import models
from training_utils import PredictionWriter


def load_pytorch_module(
        ckpt_fn: str,
        **override_hparams
):

    ckpt = torch.load(ckpt_fn, map_location="cpu")

    # load and optionally override saved hyperparameters
    hparams = ckpt["hyper_parameters"].copy()
    # override pdb_fns
    if "pdb_fns" in hparams:
        hparams["pdb_fns"] = None
    hparams.update(override_hparams)

    # instantiate model with hyperparameters
    model = models.Model[hparams["model_name"]].cls(**hparams)

    # align the state dict / model keys
    state_dict = align_state_dict_keys(
        ckpt["state_dict"], set(model.state_dict().keys())
    )

    # Load weights
    model.load_state_dict(state_dict, strict=True)

    return model


class PredictModelSummary(ModelSummary):
    """ the standard ModelSummary callback only triggers on fit start
        hence this callback which triggers on predict start """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_predict_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule") -> None:
        self.on_fit_start(trainer, pl_module)


def find_state_dict_transform(
        checkpoint_keys: set[str],
        model_keys: set[str]
) -> tuple[Optional[str], Optional[str]]:

    """ this assumes the only mismatch between the checkpoint and model keys
        is due to uniform prefixing (like model., model.model., etc.) and that
        the checkpoint and the model keys follow the same class structure but
        were saved with different wrappers...

        this will work for the existing models trained in this repository
    """

    # make sure we are getting sets (not dict_keys) so we can do set operations
    checkpoint_keys = set(checkpoint_keys)
    model_keys = set(model_keys)

    # if the keys already match, no transform is needed
    if checkpoint_keys == model_keys:
        return "", ""

    # find the common prefix between the checkpoint and model keys
    # this loops through all pairs, but under our assumptions, we could
    # just check the first pair... however, overhead is minimal
    for ckpt_key in checkpoint_keys:
        for model_key in model_keys:

            if ckpt_key.endswith(model_key):
                # checkpoint key ends with model key
                # need to strip the prefix from the ckpt key to match the model key
                strip_prefix = ckpt_key[: -len(model_key)]
                # check if this transformation works
                transformed_keys = {k[len(strip_prefix):] for k in checkpoint_keys}
                if transformed_keys == model_keys:
                    return strip_prefix, ""

            elif model_key.endswith(ckpt_key):
                # model key ends with checkpoint key
                # need to add the prefix to the ckpt key to match the model key
                add_prefix = model_key[: -len(ckpt_key)]
                # check if this transformation works
                transformed_keys = {f"{add_prefix}{k}" for k in checkpoint_keys}
                if transformed_keys == model_keys:
                    return "", add_prefix

    return None, None


def transform_state_dict(
    state_dict: dict[str, Any],
    strip_prefix: str = "",
    add_prefix: str = ""
) -> dict[str, Any]:

    new_state_dict = {}
    for k, v in state_dict.items():
        if strip_prefix:
            if not k.startswith(strip_prefix):
                raise ValueError(f"Key '{k}' does not start with prefix '{strip_prefix}'")
            k = k[len(strip_prefix):]
        new_key = f"{add_prefix}{k}"
        new_state_dict[new_key] = v
    return new_state_dict


def align_state_dict_keys(
    checkpoint_state_dict: dict[str, Any],
    model_state_dict_keys: set[str],
    *,
    warn: bool = True
) -> dict[str, Any]:

    ckpt_keys = set(checkpoint_state_dict.keys())
    strip_prefix, add_prefix = find_state_dict_transform(ckpt_keys, model_state_dict_keys)

    if (strip_prefix, add_prefix) == (None, None):
        raise RuntimeError("Unable to match checkpoint keys to model keys exactly.")

    if (strip_prefix, add_prefix) == ("", ""):
        return checkpoint_state_dict  # No change needed

    if warn:
        warnings.warn(
            f"Transforming checkpoint keys: "
            f"strip_prefix='{strip_prefix}', add_prefix='{add_prefix}'"
        )

    return transform_state_dict(checkpoint_state_dict, strip_prefix, add_prefix)


def load_lightning_module(
    cls: Type,
    ckpt: dict,
    strict: bool = True,
    **override_hparams
):

    # Load and optionally override saved hyperparameters
    hparams = ckpt["hyper_parameters"].copy()
    hparams.update(override_hparams)

    # Instantiate model with hyperparameters
    model = cls(**hparams)

    # Align the state dict / model keys
    state_dict = align_state_dict_keys(
        ckpt["state_dict"], set(model.state_dict().keys())
    )

    # Load weights
    model.load_state_dict(state_dict, strict=strict)

    return model


def main(args):

    # determine whether we are loading a DMSTask or RosettaTask checkpoint
    # based on the presence of the 'save_hyperparams' argument in the checkpoint
    # this is not an ideal method of determining the task, but it works because
    # the DMSTask class has a 'save_hyperparams' argument, while RosettaTask does not
    # todo: add placeholder arguments to both classes specifically for this purpose
    ckpt = torch.load(args.pretrained_ckpt_path, map_location="cpu")
    if "save_hyperparams" in ckpt["hyper_parameters"]:
        # lm = tasks.DMSTask.load_from_checkpoint(args.pretrained_ckpt_path, pdb_fns=None)
        task_class = tasks.DMSTask
    else:
        # lm = tasks.RosettaTask.load_from_checkpoint(args.pretrained_ckpt_path, pdb_fns=None)
        task_class = tasks.RosettaTask

    # using a custom checkpoint loading solution which supports the "converted"
    # model checkpoints found in the metl-pretrained repo
    # the downside is that it doesn't load optimizer states, lr schedulers, etc.
    # if we want to load those, we can use the load_from_checkpoint methods above
    lm = load_lightning_module(
        task_class,
        ckpt,
        pdb_fns=None,  # will be handled during runtime
    )

    # validation that applies to both dms and rosetta datamodules
    # check if the provided encoding matches the hyperparameters
    # special case to handle old rosetta checkpoints where "auto" just meant "int_seqs"
    ckpt_encoding = "int_seqs" if lm.hparams.encoding == "auto" else lm.hparams.encoding
    if args.encoding != ckpt_encoding:
        if not args.override_encoding:
            warnings.warn(f"the given encoding ({args.encoding}) does not match the "
                          f"checkpoint hyperparams ({ckpt_encoding}). setting "
                          f"the encoding to {ckpt_encoding}. to override this, "
                          f"provide the --override_encoding flag.")
            args.encoding = ckpt_encoding
        else:
            print(f"overriding encoding in checkpoint ({lm.hparams.encoding}) with "
                  f"provided encoding ({args.encoding})")

    # load the datamodule
    if args.dataset_type == "dms":
        if args.predict_mode == "all_sets":
            warnings.warn("predict_mode 'all_sets' is not currently supported "
                          "for inference. setting it to 'full_dataset'.")
            args.predict_mode = "full_dataset"
        dm = datamodules.DMSDataModule(**vars(args))
    elif args.dataset_type == "rosetta":
        dm = datamodules.BasicRosettaDataModule(**vars(args))
    else:
        raise ValueError("Dataset type must be either 'dms' or 'rosetta'")

    # update the example input array now that the datamodule is loaded
    lm.example_input_array = dm.example_input_array

    # determine the output directory and save file
    uuid = lm.hparams.uuid

    if args.dataset_type == "dms":
        if args.wt:
            output_dir = join(args.log_dir_base, f"{uuid}/wt_{args.ds_name}")
        else:
            output_dir = join(args.log_dir_base, f"{uuid}/dms_{args.ds_name}")

    elif args.dataset_type == "rosetta":
        # the rosetta ds name is the name of the directory that args.rosetta_db_fn is in
        rosetta_ds_name = basename(dirname(args.ds_fn))
        # if a split_dir is specified, and we are predicting a set from the split dir,
        # include the split dir in the directory structure
        if args.split_dir is not None:
            output_dir = join(
                args.log_dir_base,
                f"{uuid}/rosetta_{rosetta_ds_name}/{basename(args.split_dir)}/{args.predict_mode}"
            )
        else:
            output_dir = join(
                args.log_dir_base,
                f"{uuid}/rosetta_{rosetta_ds_name}/{args.predict_mode}"
            )
    else:
        raise ValueError("Dataset type must be either 'dms' or 'rosetta'")

    if args.run_dir is not None:
        output_dir = join(args.run_dir, output_dir)

    print("Output directory: {}".format(output_dir))

    # check if the output file already exists skip if it does
    save_fn_base = "predictions"

    # check if the epoch-level output file already exists
    # save file for the epoch write (just the save_fn_base with .npy extension)
    save_fn_epoch = join(output_dir, f"{save_fn_base}.npy")
    if isfile(save_fn_epoch):
        print("Output file {} already exists, skipping".format(save_fn_epoch))
        return

    print("Writing predictions to {}".format(save_fn_epoch))

    # prediction writer callback to save predictions to file
    pred_writer = PredictionWriter(output_dir=output_dir,
                                   save_fn_base=save_fn_base,
                                   batch_write_mode=args.batch_write_mode,
                                   write_interval=args.write_interval)

    # number of GPUs for trainer
    accelerator = "cpu"
    devices = "auto"
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif mps_available:
        accelerator = "mps"
        devices = 1

    callbacks: list[Callback] = [pred_writer]
    if args.show_model_summary:
        callbacks.append(PredictModelSummary(max_depth=-1))

    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices
    )
    trainer.predict(lm, datamodule=dm, return_predictions=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # prediction settings
    parser.add_argument(
        "--write_interval",
        type=str,
        help="write interval for predictions",
        choices=["batch", "epoch", "batch_and_epoch"],
        default="batch"
    )
    parser.add_argument(
        "--batch_write_mode",
        type=str,
        help="batch write mode for predictions",
        choices=["combined_csv", "separate_csv", "separate_npy"],
        default="separate_csv"
    )

    # model information
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=str,
        help="path to checkpoint",
        required=True
    )
    parser.add_argument(
        "--override_encoding",
        action="store_true",
        help="override encoding in checkpoint"
    )

    # the type of dataset we are running inference for
    parser.add_argument(
        "--dataset_type",
        type=str,
        help="this script supports both dms and rosetta datasets",
        choices=["dms", "rosetta"],
        default="dms"
    )

    parser.add_argument(
        "--wt",
        action="store_true",
        help="prediction for wild-type variant only"
    )

    # misc
    parser.add_argument("--show_model_summary", action="store_true")
    parser.add_argument(
        "--run_dir",
        type=str,
        help="run directory, prepended to output_dir, for compat with local runs",
        default=None
    )
    parser.add_argument(
        "--log_dir_base",
        type=str,
        help="output directory",
        default="output/inference"
    )

    # add cluster, process, and github_tag args, which are ignored by this script
    # needed solely for compatability with htcondor runs
    # maybe in the future we could at least make this script log these arguments
    parser.add_argument("--cluster", type=str, default=None)
    parser.add_argument("--process", type=str, default=None)
    parser.add_argument("--github_tag", type=str, default=None)

    # First stage: parse known args to decide dataset type
    partial_args, _ = parser.parse_known_args()

    # Dynamically extend parser
    if partial_args.dataset_type == "rosetta":
        parser = datamodules.BasicRosettaDataModule.add_data_specific_args(parser)
    elif partial_args.dataset_type == "dms":
        parser = datamodules.DMSDataModule.add_data_specific_args(parser)

    parsed_args = parser.parse_args()

    main(parsed_args)
