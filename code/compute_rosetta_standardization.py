""" compute standardization parameters for rosetta datasets """

import os
from os.path import join, dirname, isfile
from typing import Optional, cast
import argparse
import logging

import pandas as pd

try:
    from . import split_dataset as sd
except ImportError:
    import split_dataset as sd

logger = logging.getLogger("METL." + __name__)
logger.setLevel(logging.DEBUG)


def save_standardize_params(ds_fn: str,
                            split_dir: Optional[str] = None,
                            energies_start_col: str = "total_score",
                            columns2ignore:Optional[list[str]]=None):
    """ save the means and standard deviations of all rosetta energies in dataset.
        if there are multiple different PDBs, such as in the global rosetta dataset,
        then the means and standard deviations are computed for each PDB separately """

    ds = cast(pd.DataFrame, pd.read_hdf(ds_fn, key="variant"))

    print('ds columns')
    print(ds.columns)
    # Drop user-specified columns if they exist
    if columns2ignore is None:
        pass
    else:
        ds = ds.drop(columns=[col for col in columns2ignore if col in ds.columns])
    print('ds columns')
    print(ds.columns)


    # default output directory for full-dataset standardization params
    out_dir = join(dirname(ds_fn), "standardization_params")
    out_suffix = "all"

    # if params are being calculated on just the training set, grab a dataframe of just the training set
    # and set the output directory to the split directory because the params will be specific to this split
    if split_dir is not None:
        # these params will be specific to this split
        out_dir = join(split_dir, "standardization_params")
        out_suffix = "train"

        # given a split dir, so only compute the standardization parameters on the train set
        set_idxs = sd.load_split_dir(split_dir)["train"]
        ds = ds.iloc[set_idxs]
        logger.info("computing standardization params on training set only")
    else:
        logger.info("computing standardization params on full dataset")

    # ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    logger.info("saving standardization params to: {}".format(out_dir))

    # standardization parameters are computed per-pdb
    g = ds.groupby("pdb_fn")
    g_mean = g.mean(numeric_only=True)
    g_mean = g_mean.iloc[:, list(g_mean.columns).index(energies_start_col):]

    # ddof=0 to match sklearn's StandardScaler (for a biased estimator of standard deviation)
    g_std = g.std(ddof=0, numeric_only=True)
    g_std = g_std.iloc[:, list(g_std.columns).index(energies_start_col):]

    means_out_fn = join(out_dir, "energy_means_{}.tsv".format(out_suffix))
    stds_out_fn = join(out_dir, "energy_stds_{}.tsv".format(out_suffix))
    if isfile(means_out_fn) or isfile(stds_out_fn):
        raise FileExistsError(
            "Standardization params output file(s) already exist: {} or {}".format(means_out_fn, stds_out_fn))

    g_mean.to_csv(means_out_fn, sep="\t", float_format="%.7f")
    g_std.to_csv(stds_out_fn, sep="\t", float_format="%.7f")


def main(args):
    save_standardize_params(args.ds_fn_h5, args.split_dir, args.energies_start_col,args.columns2ignore)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")


    parser.add_argument("ds_fn_h5",
                        help="path to the rosetta dataset in hdf5 format",
                        type=str)
    parser.add_argument("--split_dir",
                        help="path to the split directory containing the train/val/test split indices. if provided, "
                             "the standardization parameters will be computed on the training set only. this is "
                             "necessary for training a source model.",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--energies_start_col",
                        help="the column name of the first energy term in the dataset. default is 'total_score'. "
                             "this is used to determine which columns in the dataset are energy terms. "
                             "leave this as default unless for some reason total_score is not the first energy term.",
                        type=str,
                        default="total_score",
                        required=False)

    parser.add_argument("--columns2ignore",
                        help="if their is not clear indication of which energy term comes first (such as when"
                        "combining energy terms from mutliple Rosetta energy functions) use this parameter to"
                             " define columns which must be ignored",
                        type=str,
                        nargs='+',
                        default="total_score",
                        required=False)

    main(parser.parse_args())