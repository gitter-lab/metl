""" Run inference with METL models."""

import argparse
import warnings
from os.path import join, isfile, basename, dirname
from typing import Optional, Any

import pandas as pd
import pytorch_lightning as pl
import torch
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
from training_utils import PredictionWriter


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


def get_ident(uuid, uuid_ident_map_fn="output/model_index/uuid_ident_map.csv"):
    ident = None
    # check if the uuid_ident_map file exists
    if isfile(uuid_ident_map_fn):
        uuid_ident_map = pd.read_csv(uuid_ident_map_fn)
        if uuid in uuid_ident_map.uuid.values:
            ident = uuid_ident_map[uuid_ident_map.uuid == uuid].ident.values[0]
    return ident


def find_state_dict_prefix(state_dict, model_keys):
    """
    Finds the correct prefix to transform checkpoint keys to match model keys exactly.
    Assumes model expects 'model.model.' keys, and checkpoint keys are either:
    - 'model.model.' (as-is, no change needed)
    - 'model.' (requires prefixing with 'model.' because checkpoint was converted to be
                more compatible with pure PyTorch / for metl-pretrained)
    """
    prefixes = ["", "model."]  # try no prefix, or add 'model.' to each key
    model_keys_set = set(model_keys)

    for prefix in prefixes:
        if prefix:
            transformed_keys = {f"{prefix}{k}" for k in state_dict}
        else:
            transformed_keys = set(state_dict)

        if transformed_keys == model_keys_set:
            return prefix
    return None


def load_lightning_module(cls, ckpt, strict=True, **override_hparams):
    # Load and optionally override saved hyperparameters
    hparams = ckpt["hyper_parameters"].copy()
    hparams.update(override_hparams)

    # Instantiate model with hyperparameters
    model = cls(**hparams)

    # Determine correct prefix to apply (if any)
    prefix = find_state_dict_prefix(ckpt["state_dict"], model.state_dict().keys())
    if prefix is None:
        raise RuntimeError("Unable to match checkpoint keys to model keys exactly.")

    # Apply prefix if needed
    if prefix:
        warnings.warn(f"Prefixing checkpoint keys with '{prefix}' to match model format.")
        state_dict = {f"{prefix}{k}": v for k, v in ckpt["state_dict"].items()}
    else:
        state_dict = ckpt["state_dict"]

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

    # load the datamodule
    if args.dataset_type == "dms":
        if args.predict_mode == "all_sets":
            warnings.warn("predict_mode 'all_sets' is not currently supported "
                          "for inference. setting it to 'full_dataset'.")
            args.predict_mode = "full_dataset"

        # check if the provided encoding matches the hyperparameters
        if args.encoding != lm.hparams.encoding:
            if not args.override_encoding:
                warnings.warn(f"the given encoding ({args.encoding}) does not match the "
                              f"checkpoint hyperparams ({lm.hparams.encoding}). setting "
                              f"the encoding to {lm.hparams.encoding}. to override this, "
                              f"provide the --override_encoding flag.")
                args.encoding = lm.hparams.encoding
            else:
                print(f"overriding encoding in checkpoint ({lm.hparams.encoding}) with "
                      f"provided encoding ({args.encoding})")

        dm = datamodules.DMSDataModule(**vars(args))

    elif args.dataset_type == "rosetta":
        dm = datamodules.BasicRosettaDataModule(
            ds_fn=args.rosetta_db_fn,
            split_dir=args.rosetta_split_dir,
            predict_mode=args.rosetta_predict_mode,
            batch_size=args.batch_size,
            encoding=args.encoding
        )
    else:
        raise ValueError("Dataset type must be either 'dms' or 'rosetta'")

    # update the example input array now that the datamodule is loaded
    lm.example_input_array = dm.example_input_array

    # determine the output directory and save file
    uuid = lm.hparams.uuid
    ident = get_ident(uuid)

    if args.dataset_type == "dms":
        if args.wt:
            output_dir = join(args.log_dir_base, "{}_{}/wt_{}".format(ident, uuid, args.ds_name))
        else:
            output_dir = join(args.log_dir_base, "{}_{}/dms_{}".format(ident, uuid, args.ds_name))

    elif args.dataset_type == "rosetta":
        # the rosetta ds name is the name of the directory that args.rosetta_db_fn is in
        rosetta_ds_name = basename(dirname(args.rosetta_db_fn))
        # if a split_dir is specified, and we are predicting a set from the split dir, we want to
        # include the split dir in the directory structure
        if args.rosetta_split_dir is not None:
            output_dir = join(
                args.log_dir_base,
                "{}_{}/rosetta_{}/{}/{}".format(
                    ident,
                    uuid,
                    rosetta_ds_name,
                    basename(args.rosetta_split_dir),
                    args.rosetta_predict_mode))
        else:
            output_dir = join(
                args.log_dir_base,
                "{}_{}/rosetta_{}/{}".format(
                    ident,
                    uuid,
                    rosetta_ds_name,
                    args.rosetta_predict_mode))
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

    trainer = pl.Trainer(logger=False,
                         callbacks=[pred_writer, PredictModelSummary(max_depth=-1)],
                         accelerator=accelerator,
                         devices=devices)
    trainer.predict(lm, datamodule=dm, return_predictions=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # prediction settings
    # batch_write_mode: str = "combined_csv",
    # write_interval: str = "batch_and_epoch"
    parser.add_argument("--write_interval", type=str, help="write interval for predictions",
                        choices=["batch", "epoch", "batch_and_epoch"], default="batch")
    parser.add_argument("--batch_write_mode", type=str, help="batch write mode for predictions",
                        choices=["combined_csv", "separate_csv", "separate_npy"], default="separate_csv")

    # model information
    parser.add_argument("--task", type=str, help="Rosetta or DMS task", choices=["rosetta", "dms"])
    parser.add_argument("--pretrained_ckpt_path", type=str, help="path to checkpoint", required=True)
    parser.add_argument("--override_encoding", action="store_true", help="override encoding in checkpoint")

    # the type of dataset we are running inference for
    parser.add_argument("--dataset_type", type=str,
                        help="this script supports both dms and rosetta datasets",
                        choices=["dms", "rosetta"], default="dms")

    # # dms dataset type
    # parser.add_argument("--ds_name", type=str, help="name of the dms dataset for the dms dataset type")
    parser.add_argument("--wt", action="store_true", help="prediction for wild-type variant only")

    # rosetta dataset type
    parser.add_argument("--rosetta_db_fn", type=str, help="database file for rosetta type datasets")
    parser.add_argument("--rosetta_predict_mode", type=str, help="predict mode for rosetta type datasets")
    parser.add_argument("--rosetta_split_dir", type=str, help="split dir for rosetta type datasets")

    # misc
    # parser.add_argument("--batch_size", type=int, help="batch size", default=128)
    parser.add_argument("--run_dir", type=str,
                        help="run directory, prepended to output_dir, for compat with local runs simulating",
                        default=None)
    parser.add_argument("--log_dir_base", type=str, help="output directory", default="output/inference")

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
        parser = datamodules.RosettaDataModule.add_data_specific_args(parser)
    elif partial_args.dataset_type == "dms":
        parser = datamodules.DMSDataModule.add_data_specific_args(parser)

    parsed_args = parser.parse_args()

    main(parsed_args)



