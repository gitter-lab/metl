import argparse
import os
from os.path import join
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

try:
    from . import datamodules
    from .training_utils import create_log_dir, save_metrics_custom, save_scatterplots
    from . import utils
except ImportError:
    import datamodules
    from training_utils import create_log_dir, save_metrics_custom, save_scatterplots
    import utils


class SKDMSDataset:
    """ simple wrapper for variants, encoded_seqs, and targets """
    def __init__(self, variants, encoded_seqs, targets):
        self.variants = variants
        self.encoded_seqs = encoded_seqs
        self.targets = targets

    def __str__(self):
        return "<SKDMSDataset num_examples: {}>".format(len(self.variants))


class SKDMSDataModule(datamodules.DMSDataModule):
    """ slightly modified DMSDataModule to return an SKDMSDataset instead the METL dataset """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ds(self, set_name: Optional[str] = None):
        variants = self.get_variants(set_name)
        targets = self.get_targets(set_name)
        enc_data = self.get_encoded_data(set_name)

        return SKDMSDataset(variants=variants, encoded_seqs=enc_data, targets=targets)


def save_predictions(predictions_d, log_dir):
    predictions_dir = join(log_dir, "predictions")
    utils.mkdir(predictions_dir)

    for set_name, set_preds in predictions_d.items():
        np.savetxt(join(predictions_dir, f"{set_name}_predictions.txt"), set_preds, fmt="%1.7f", delimiter=",")
        # np.save(join(predictions_dir, f"{set_name}_predictions.npy"), set_preds)


def get_predictions(model, train_ds, val_ds, test_ds, dm):
    train_predictions = model.predict(train_ds.encoded_seqs).flatten()
    val_predictions = None if val_ds is None else model.predict(val_ds.encoded_seqs).flatten()
    test_predictions = model.predict(test_ds.encoded_seqs).flatten()
    return {"train": train_predictions,
            "val": val_predictions,
            "test": test_predictions}


def main(args):

    # get the uuid and log directory for this run
    log_dir_base = args.log_dir_base
    if args.run_dir is not None:
        log_dir_base = join(args.run_dir, log_dir_base)
    my_uuid, log_dir = create_log_dir(log_dir_base, args.uuid)
    utils.save_args(vars(args), join(log_dir, "args.txt"), ignore=["cluster", "process"])

    # load datamodule
    dm = SKDMSDataModule(flatten_encoded_data=True, **vars(args))

    # get datasets from datamodule
    train_ds = dm.get_ds("train")
    val_ds = dm.get_ds("val") if dm.has_val_set else None
    test_ds = dm.get_ds("test")

    # train the model
    if args.model_name == "linear_regression":
        model = LinearRegression()
    elif args.model_name == "ridge":
        model = Ridge(solver="cholesky")
    elif args.model_name == "mlp":
        # Sklearn MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(1024,),
                             activation="relu",
                             solver="adam",
                             alpha=0,
                             batch_size=128,
                             learning_rate="constant",
                             learning_rate_init=0.001,
                             max_iter=500,
                             shuffle=True)
    else:
        raise ValueError("unsupported model_name: {}".format(args.model_name))

    model.fit(train_ds.encoded_seqs, train_ds.targets)

    # compute predictions for train, val, and test sets
    predictions_d = get_predictions(model, train_ds, val_ds, test_ds, dm)

    # save predictions
    save_predictions(predictions_d, log_dir)

    # save metrics and scatterplots
    save_metrics_custom(dm, predictions_d, log_dir)
    if args.save_scatterplots:
        save_scatterplots(dm, predictions_d, log_dir)

    if not args.delete_checkpoints:
        # save model to disk
        checkpoints_dir = join(log_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        joblib.dump(model, join(checkpoints_dir, "model.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # program
    parser.add_argument("--run_dir",
                        help="run directory, prepended to log_dir_base, for compat with local runs simulating a "
                             "condor run",
                        type=str,
                        default=None)

    parser.add_argument("--save_scatterplots",
                        help="save scatterplots",
                        action="store_true")

    # Model
    parser.add_argument("--model_name",
                        help="what model to train",
                        type=str,
                        default="linear_regression",
                        choices=["linear_regression", "ridge", "mlp"])

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

    parser.add_argument("--delete_checkpoints",
                        action="store_true",
                        default=False)

    # add data specific args
    parser = SKDMSDataModule.add_data_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)
    main(parser.parse_args())

