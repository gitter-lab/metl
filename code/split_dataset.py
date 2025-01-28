""" split dataset into train, val, and test sets """

import random
import warnings
from os.path import join, isfile, isdir, basename
import os
from collections.abc import Iterable
import hashlib
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import utils
except ImportError:
    import utils


logger = logging.getLogger("METL." + __name__)
logger.setLevel(logging.DEBUG)


def supertest(ds, size=.1, rseed=8, out_dir=None, overwrite=False):
    """ create a supertest split, meant to be completely held out data until final evaluation """
    np.random.seed(rseed)
    idxs = np.arange(0, ds.shape[0])
    idxs, super_test_idxs = train_test_split(idxs, test_size=size)
    save_fn = None
    if out_dir is not None:
        utils.mkdir(out_dir)
        out_fn = "supertest_w{}_s{}_r{}.txt".format(hash_withhold(super_test_idxs), size, rseed)
        save_fn = join(out_dir, out_fn)
        if isfile(save_fn) and not overwrite:
            raise FileExistsError("supertest split already exists: {}".format(join(out_dir, out_fn)))
        else:
            logger.info("saving supertest split to file {}".format(save_fn))
            utils.save_lines(save_fn, super_test_idxs)
    return np.array(super_test_idxs, dtype=int), save_fn


def load_withhold(withhold):
    """ load indices to withhold from split (for supertest set, for example) """
    if isinstance(withhold, str):
        if not isfile(withhold):
            raise FileNotFoundError("couldn't find file w/ indices to withhold: {}".format(withhold))
        else:
            withhold = np.loadtxt(withhold, dtype=int)
    elif not isinstance(withhold, Iterable):
        raise ValueError("withhold must be a string specifying a filename containing indices to withhold "
                         "or an iterable containing those indices")
    return np.array(withhold, dtype=int)


def hash_withhold(withheld_idxs, length=6):
    """ hash the withheld indices for file & directory naming purposes """
    hash_object = hashlib.shake_256(withheld_idxs)
    w = hash_object.hexdigest(length)
    return w


def train_val_test(ds, train_size=.90, val_size=.1, test_size=0., withhold=None,
                   rseed=8, singles_only=False, out_dir=None, overwrite=False):

    """ split data into train, val, and test sets """
    if train_size + val_size + test_size != 1:
        raise ValueError("train_size, val_size, and test_size must add up to 1. current values are "
                         "tr={}, tu={}, and te={}".format(train_size, val_size, test_size))

    # set the random seed
    np.random.seed(rseed)

    # keep track of all the splits we make
    split = {}

    # set up the indices that will get split

    # if singles_only is set, filter the indices to only include single mutants
    if singles_only:
        idxs = np.where(ds["num_mutations"] == 1)[0]
    else:
        idxs = np.arange(0, ds.shape[0])

    # withhold supertest data if specified -- can be either a file specifying idxs or an iterable with idxs
    if withhold is not None:
        withhold = load_withhold(withhold)
        # the withheld indices will be saved as part of the split for future reference
        split["stest"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    if val_size > 0:
        if val_size == 1:
            split["val"] = idxs
        else:
            idxs, val_idxs = train_test_split(idxs, test_size=val_size)
            split["val"] = val_idxs
    if test_size > 0:
        adjusted_test_size = np.around(test_size / (1 - val_size), 5)
        if adjusted_test_size == 1:
            split["test"] = idxs
        else:
            idxs, test_idxs = train_test_split(idxs, test_size=adjusted_test_size)
            split["test"] = test_idxs
    if train_size > 0:
        adjusted_train_size = np.around(train_size / (1 - val_size - test_size), 5)
        if adjusted_train_size == 1:
            split["train"] = idxs
        else:
            idxs, train_idxs = train_test_split(idxs, test_size=adjusted_train_size)
            split["train"] = train_idxs

    out_dir_split = None
    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split["stest"])

        if singles_only:
            od = "standard-singles_tr{}_tu{}_te{}_w{}_r{}".format(train_size, val_size, test_size, w, rseed)
            out_dir_split = join(out_dir, od)
        else:
            od = "standard_tr{}_tu{}_te{}_w{}_r{}".format(train_size, val_size, test_size, w, rseed)
            out_dir_split = join(out_dir, od)

        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            logger.info("saving train-val-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)
    return split, out_dir_split


def resampled_dataset_size(
        # the number of examples in the full dataset
        full_dataset_size: int,
        # fraction of full dataset to use as test set
        test_fraction: float,
        # number of examples to include in the reduced dataset
        dataset_size: int,
        # fraction of dataset_size to use as validation set (rest is train set)
        val_fraction: float,
        # number of replicates
        num_replicates: int,
        # filename or iterable containing indices for super test
        withhold: Optional[Union[str, Iterable]],
        # random seed
        rseed: int = 0,
        # output directory to save split to disk
        out_dir: Optional[str] = None,
        # whether to overwrite existing split if the out_dir already exists
        overwrite: bool = False):

    """ computes the test split using the full amount of data remaining after removing the withholding.
        then from remaining data, it generates requested number of dataset replicates at the requested dataset size.
        the dataset is split into train and validation based on the val_prop. """

    # set the random seed
    np.random.seed(rseed)

    # set up the indices that will get split
    idxs = np.arange(0, full_dataset_size)

    # each replicate will have the same test split
    split_template = {}

    # withhold supertest data if specified -- can be either a file specifying idxs or an iterable with idxs
    if withhold is not None:
        withhold = load_withhold(withhold)
        # the withheld indices will be saved as part of the split for future reference
        split_template["stest"] = withhold
        # remove the idxs to withhold from the pool of idxs
        idxs = np.array(sorted(set(idxs) - set(withhold)), dtype=int)

    # add the test split to the split template (will be used across all replicate splits)
    if test_fraction > 0:
        idxs, test_idxs = train_test_split(idxs, test_size=test_fraction)
        split_template["test"] = test_idxs

    if len(idxs) < dataset_size:
        raise ValueError("Not enough remaining idxs to sample a dataset of size {}".format(dataset_size))

    # there will be num_replicates splits
    splits = []
    for i in range(num_replicates):
        # make a copy of the template containing test and withheld indices
        split_i = split_template.copy()

        # split the data into the training and validation pools (based on specified val_fraction)
        # this helps with making the splits build on each other w/ increasing dataset sizes
        # each replicate will have its own randomized pools so this is where the per-replicate randomness is from
        train_pool, val_pool = train_test_split(idxs, test_size=val_fraction)

        # determine train and val sizes based on the given dataset size and val_fraction
        val_size = np.rint(dataset_size * val_fraction).astype(int)
        train_size = dataset_size - val_size

        train_idxs = train_pool[:train_size]
        val_idxs = val_pool[:val_size]

        split_i["train"] = train_idxs
        split_i["val"] = val_idxs
        splits.append(split_i)

    # save out to directory
    out_dir_split = None
    if out_dir is not None:
        # compute a hash of the withheld indices (if any) in order to support at least some name differentiation
        w = "F" if withhold is None else hash_withhold(split_template["stest"])
        format_args = [dataset_size, val_fraction, test_fraction, w, num_replicates, rseed]
        out_dir_split = join(out_dir, "resampled_ds{}_val{}_te{}_w{}_s{}_r{}".format(*format_args))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}. if you think this is a withholding hash collision, "
                                  "i recommend increasing hash length or specifying an out_dir other than {}".format(
                                    out_dir_split, out_dir))
        else:
            logger.info("saving resampled split to directory {}".format(out_dir_split))
            for i, split in enumerate(splits):
                out_dir_split_rep = join(out_dir_split, basename(out_dir_split) + "_rep_{}".format(i))
                save_split(split, out_dir_split_rep)

    return splits, out_dir_split


def regime_split(ds, train_regimes, test_regimes, train_size, val_size, test_size,
                 rseed=10, out_dir=None, overwrite=False):

    if not all(ds.index == range(0, len(ds))):
        warnings.warn("The given ds.index is not sequential in range (0, len(ds)). "
                      "This function uses ds indices, so if the end goal is to have a split that indexes into "
                      "the given ds, make sure to ds.reset_index()")

    # make the regimes into lists if just given in
    if not isinstance(train_regimes, Iterable):
        train_regimes = [train_regimes]
    if not isinstance(test_regimes, Iterable):
        test_regimes = [test_regimes]

    if not set(train_regimes).isdisjoint(test_regimes):
        raise ValueError("Train and test regimes must be disjoint")

    # figure out what regimes are available for input validation
    available_regimes = ds["variant"].apply(lambda v: len(v.split(","))).unique()

    # check if we were given any regimes not in the dataset
    for r in train_regimes + test_regimes:
        if r not in available_regimes:
            raise ValueError("Regime {} not in dataset. Available regimes: {}".format(r, available_regimes))

    # get train regime and test regime indices
    train_pool_idxs = []
    test_pool_idxs = []
    discard_pool_idxs = []
    for idx, variant in ds["variant"].items():
        num_muts = len(variant.split(","))
        if num_muts in train_regimes:
            train_pool_idxs.append(idx)
        elif num_muts in test_regimes:
            test_pool_idxs.append(idx)
        else:
            discard_pool_idxs.append(idx)
    # convert to numpy arrays (for consistency with loading function)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)
    discard_pool_idxs = np.array(discard_pool_idxs, dtype=int)
    logger.info("train pool size: {}, test pool size: {}, discard pool size: {}".format(
        len(train_pool_idxs), len(test_pool_idxs), len(discard_pool_idxs)))

    # now use train_size, val_size, and test_size to create train, val, and test sets from the pools

    # create the actual split
    split = {}
    # split training pool into train and val sets
    if val_size > 0:
        if val_size == 1:
            split["val"] = train_pool_idxs
        else:
            train_idxs, val_idxs = train_test_split(train_pool_idxs, test_size=val_size)
            split["train"] = train_idxs
            split["val"] = val_idxs
    else:
        split["train"] = train_pool_idxs
    # the full test pool becomes the test set

    # now grab test_size from the test_pool_idxs
    if test_size == 1:
        # note the indices will be sorted because they are coming directly from test_pool_idxs
        split["test"] = test_pool_idxs
    else:
        _, test_idxs = train_test_split(test_pool_idxs, test_size=test_size)
        split["test"] = test_idxs

    num_train = 0
    num_val = 0
    num_test = len(split["test"])
    if "train" in split:
        num_train = len(split["train"])
    if "val" in split:
        num_val = len(split["val"])
    logger.info("num_train: {}, num_val: {}, num_test: {}".format(num_train, num_val, num_test))

    # create dictionaries of regimes and pool indices to save as additional info
    regimes = {"train": train_regimes, "test": test_regimes}
    pool_dataset_idxs = {"train": train_pool_idxs, "test": test_pool_idxs, "discard": discard_pool_idxs}

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        out_dir_split = join(out_dir, "regime_tr-reg{}_te-reg{}_tr{}_tu{}_te{}_r{}".format(
            "-".join(list(map(str, train_regimes))), "-".join(list(map(str, test_regimes))), train_size, val_size, test_size, rseed))

        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-val-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)

            # save additional info such as the pool sequence positions and the pool dataset idxs
            save_split(regimes, join(out_dir_split, "additional_info", "regimes"))
            save_split(pool_dataset_idxs, join(out_dir_split, "additional_info", "pool_dataset_idxs"))

    additional_info = {"regimes": regimes, "pool_dataset_idxs": pool_dataset_idxs}

    return split, out_dir_split, additional_info


def position_split(ds: pd.DataFrame,
                   # number of sequence positions
                   seq_len: int,
                   # offset for mutation position, in case dataset variants are not 0-indexed
                   wt_ofs: int,
                   # fraction of sequence positions to use as training (1-train_pos_size will be used as test)
                   train_pos_size: float,
                   # fraction of training samples to use for tuning set (# of training samples depends on how many
                   # variants end up in the training pool...)
                   val_size: float,
                   # whether to resample the dataset to make it smaller for faster training runs
                   resample_dataset_size: Optional[int] = None,
                   rseed: int = 8,
                   out_dir: Optional[str] = None,
                   overwrite: bool = False):

    # set the random seed
    np.random.seed(rseed)

    if not all(ds.index == range(0, len(ds))):
        warnings.warn("The given ds.index is not sequential in range (0, len(ds)). "
                      "This function uses ds indices, so if the end goal is to have a split that indexes into "
                      "the given ds, make sure to ds.reset_index()")

    if train_pos_size >= 1 or train_pos_size <= 0:
        raise ValueError("train_pos_size must be in range (0, 1)")

    if val_size > 1 or val_size < 0:
        raise ValueError("val_size must be in range [0, 1] ")

    # resample the dataset to smaller size to make it for faster training runs
    if resample_dataset_size is not None:
        # note ignore_index=False is important because we rely on the pandas index
        # to get the actual index into the full dataset, not just the resampled dataset
        ds = ds.sample(resample_dataset_size, ignore_index=False)

    # determine the number of sequence positions that will be train-set only
    num_train_positions = int(np.round(seq_len * train_pos_size))
    num_test_positions = int(seq_len - num_train_positions)
    logger.info("num_train_positions: {}, num_test_positions: {}".format(num_train_positions, num_test_positions))
    if num_train_positions == 0 or num_test_positions == 0:
        raise RuntimeError("num_train_positions and num_test_positions can't be zero")

    # determine which sequence positions will be marked train and which will be marked test
    position_idxs = np.arange(0, seq_len)
    train_positions, test_positions = train_test_split(position_idxs, train_size=num_train_positions)

    # find training, test, and overlapping pools of variants
    train_pool_idxs = []
    test_pool_idxs = []
    overlap_pool_idxs = []
    for idx, variant in ds["variant"].items():
        muts = variant.split(",")
        mut_positions = [int(mut[1:-1]) - wt_ofs for mut in muts]
        if all(mut_pos in train_positions for mut_pos in mut_positions):
            # train variant
            train_pool_idxs.append(idx)
        elif all(mut_pos in test_positions for mut_pos in mut_positions):
            # test variant
            test_pool_idxs.append(idx)
        else:
            overlap_pool_idxs.append(idx)
    # convert to numpy arrays (for consistency with loading function)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)
    overlap_pool_idxs = np.array(overlap_pool_idxs, dtype=int)
    logger.info("train pool size: {}, test pool size: {}, overlap pool size: {}".format(
        len(train_pool_idxs), len(test_pool_idxs), len(overlap_pool_idxs)))

    # create the actual split
    split = {}
    # split training pool into train and val sets
    if val_size > 0:
        if val_size == 1:
            split["val"] = train_pool_idxs
        else:
            train_idxs, val_idxs = train_test_split(train_pool_idxs, test_size=val_size)
            split["train"] = train_idxs
            split["val"] = val_idxs
    else:
        split["train"] = train_pool_idxs
    # the full test pool becomes the test set
    # note: the test set idxs will be sorted because they are coming directly from the test_pool_idxs
    split["test"] = test_pool_idxs

    num_train = 0
    num_val = 0
    num_test = len(split["test"])
    if "train" in split:
        num_train = len(split["train"])
    if "val" in split:
        num_val = len(split["val"])
    logger.info("num_train: {}, num_val: {}, num_test: {}".format(num_train, num_val, num_test))

    # create dictionaries of pool sequence positions and pool dataset idxs to save and return
    pool_seq_positions = {"train": train_positions, "test": test_positions}
    pool_dataset_idxs = {"train": train_pool_idxs, "test": test_pool_idxs, "overlap": overlap_pool_idxs}

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        if resample_dataset_size is None:
            out_dir_split = join(out_dir, "position_tr-pos{}_tu{}_r{}".format(train_pos_size, val_size, rseed))
        else:
            out_dir_split = join(out_dir, "position_tr-pos{}_tu{}_resample{}_r{}".format(train_pos_size, val_size,
                                                                                         resample_dataset_size, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-val-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)

            # save additional info such as the pool sequence positions and the pool dataset idxs
            save_split(pool_seq_positions, join(out_dir_split, "additional_info", "pool_seq_positions"))
            save_split(pool_dataset_idxs, join(out_dir_split, "additional_info", "pool_dataset_idxs"))

    additional_info = {"pool_seq_positions": pool_seq_positions, "pool_dataset_idxs": pool_dataset_idxs}

    return split, out_dir_split, additional_info


def score_extrapolation_split(ds: pd.DataFrame,
                              score_name: str,
                              wt_score: float,
                              val_size: float,
                              resample_dataset_size: Optional[int] = None,
                              rseed: int = 0,
                              out_dir: Optional[str] = None,
                              overwrite: bool = False):

    if not all(ds.index == range(0, len(ds))):
        warnings.warn("The given ds.index is not sequential in range (0, len(ds)). "
                      "This function uses ds indices, so if the end goal is to have a split that indexes into "
                      "the given ds, make sure to ds.reset_index()")

    # set the random seed
    np.random.seed(rseed)

    # resample the dataset to smaller size to make it for faster training runs
    if resample_dataset_size is not None:
        ds = ds.sample(resample_dataset_size, ignore_index=False)

    # create train and test pools based on whether variant scores are greater or less than WT
    train_pool_idxs = []
    test_pool_idxs = []
    for idx, score in ds[score_name].items():
        if score <= wt_score:
            train_pool_idxs.append(idx)
        else:
            test_pool_idxs.append(idx)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)

    logger.info("train pool size: {}, test pool size: {}".format(len(train_pool_idxs), len(test_pool_idxs)))

    # create the actual split
    train_idxs, val_idxs = train_test_split(train_pool_idxs, test_size=val_size)
    test_idxs = test_pool_idxs
    split = {"train": train_idxs,
             "val": val_idxs,
             "test": test_idxs}

    logger.info("num_train: {}, num_val: {}, num_test: {}".format(len(train_idxs), len(val_idxs), len(test_idxs)))

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        if resample_dataset_size is None:
            out_dir_split = join(out_dir, "score_thresh{}_tu{}_r{}".format(
                wt_score, val_size, rseed
            ))
        else:
            out_dir_split = join(out_dir, "score_thresh{}_tu{}_resample{}_r{}".format(
                wt_score, val_size, resample_dataset_size, rseed
            ))

        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-val-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)

    return split, out_dir_split


def mutation_split(ds: pd.DataFrame,
                   train_muts_size: float,
                   val_size: float,
                   # whether to resample the dataset to make it smaller for faster training runs
                   resample_dataset_size: Optional[int] = None,
                   rseed: int = 8,
                   out_dir: Optional[str] = None,
                   overwrite: bool = False):

    if not all(ds.index == range(0, len(ds))):
        warnings.warn("The given ds.index is not sequential in range (0, len(ds)). "
                      "This function uses ds indices, so if the end goal is to have a split that indexes into "
                      "the given ds, make sure to ds.reset_index()")

    # set the random seed
    np.random.seed(rseed)

    # resample the dataset to smaller size to make it for faster training runs
    if resample_dataset_size is not None:
        ds = ds.sample(resample_dataset_size, ignore_index=False)

    # all individual mutations in the dataset
    all_mutations = set()
    for variant in ds["variant"]:
        muts = variant.split(",")
        for mut in muts:
            all_mutations.add(mut)
    all_mutations = list(all_mutations)
    num_unique_mutations = len(all_mutations)
    logger.info("number of unique mutations in ds: {}".format(num_unique_mutations))

    # determine the number of mutations positions that will be train-set only
    num_train_mutations = int(np.round(num_unique_mutations * train_muts_size))
    num_test_mutations = int(num_unique_mutations - num_train_mutations)
    logger.info("num_train_mutations: {}, num_test_mutations: {}".format(num_train_mutations, num_test_mutations))

    # sample correct number of train and test mutations
    train_mutations, test_mutations = train_test_split(all_mutations, train_size=num_train_mutations)

    # find training, test, and overlapping pools of variants
    train_pool_idxs = []
    test_pool_idxs = []
    overlap_pool_idxs = []
    for idx, variant in ds["variant"].items():
        muts = variant.split(",")
        if all(mut in train_mutations for mut in muts):
            train_pool_idxs.append(idx)
        elif all(mut in test_mutations for mut in muts):
            test_pool_idxs.append(idx)
        else:
            overlap_pool_idxs.append(idx)

    # convert to numpy arrays (for consistency with loading function)
    train_pool_idxs = np.array(train_pool_idxs, dtype=int)
    test_pool_idxs = np.array(test_pool_idxs, dtype=int)
    overlap_pool_idxs = np.array(overlap_pool_idxs, dtype=int)
    logger.info("train pool size: {}, test pool size: {}, overlap pool size: {}".format(
        len(train_pool_idxs), len(test_pool_idxs), len(overlap_pool_idxs)))

    # create the actual split
    split = {}
    # split training pool into train and val sets
    if val_size > 0:
        if val_size == 1:
            split["val"] = train_pool_idxs
        else:
            train_idxs, val_idxs = train_test_split(train_pool_idxs, test_size=val_size)
            split["train"] = train_idxs
            split["val"] = val_idxs
    else:
        split["train"] = train_pool_idxs
    # the full test pool becomes the test set
    split["test"] = test_pool_idxs

    num_train = 0
    num_val = 0
    num_test = len(split["test"])
    if "train" in split:
        num_train = len(split["train"])
    if "val" in split:
        num_val = len(split["val"])
    logger.info("num_train: {}, num_val: {}, num_test: {}".format(num_train, num_val, num_test))

    # create dictionaries of pool mutations & dataset idxs
    pool_muts = {"train": train_mutations, "test": test_mutations}
    pool_dataset_idxs = {"train": train_pool_idxs, "test": test_pool_idxs, "overlap": overlap_pool_idxs}

    # save split to disk
    out_dir_split = None
    if out_dir is not None:
        if resample_dataset_size is None:
            out_dir_split = join(out_dir, "mutation_tr-muts{}_tu{}_r{}".format(train_muts_size, val_size, rseed))
        else:
            out_dir_split = join(out_dir, "mutation_tr-muts{}_tu{}_resample{}_r{}".format(train_muts_size, val_size,
                                                                                          resample_dataset_size, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving train-val-test split to directory {}".format(out_dir_split))
            # save the main split
            save_split(split, out_dir_split)
            # save additional info
            save_split(pool_muts, join(out_dir_split, "additional_info", "pool_muts"))
            save_split(pool_dataset_idxs, join(out_dir_split, "additional_info", "pool_dataset_idxs"))

    additional_info = {"pool_muts": pool_muts, "pool_dataset_idxs": pool_dataset_idxs}

    return split, out_dir_split, additional_info


def save_split(split, d):
    """ save a split to a directory """
    utils.mkdir(d)
    for k, v in split.items():
        out_fn = join(d, "{}.txt".format(k))
        utils.save_lines(out_fn, v)


def load_single_split_dir(split_dir, content="idxs", filetype="txt"):
    ignore_dirs = ["additional_info", "standardization_params"]
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if
           f not in ignore_dirs and not f.startswith(".") and f.endswith(".{}".format(filetype))]
    split = {}
    for f in fns:
        # logger.info("loading split from: {}".format(f))
        split_name = basename(f)[:-4]
        if content == "idxs":
            if filetype == "txt":
                split_data = pd.read_csv(f, header=None)[0].to_numpy()
            elif filetype == "npy":
                split_data = np.load(f)
            else:
                raise ValueError("unsupported filetype for split")
        elif content == "txt":
            split_data = utils.load_lines(f)
        else:
            raise ValueError("unsupported content type for split")
        split[split_name] = split_data
    return split


def load_split_dir(split_dir, filetype="txt"):
    """ load saved splits. has an exception for an "additional_info" directory and standardization_params directory,
        but otherwise assumes the given directory contains only text files (for a regular train-test-split)
        or only directories (containing replicates for a reduced train size split).
        any split dirs created with this script should be fine. """

    if not isdir(split_dir):
        raise FileNotFoundError("split directory doesn't exist: {}".format(split_dir))

    # get all the files in the given split_dir
    ignore_dirs = ["additional_info", "standardization_params"]
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if f not in ignore_dirs]

    # all directories... reduced split with multiple replicates
    # be sure to sort by the replicate number in ascending order
    if all(isdir(fn) for fn in fns):
        # todo: it would be better to return a dictionary of filename: split or replicate number: split
        fns = sorted(fns, key=lambda x: int(basename(x).split("_")[-1]))
        splits = []
        for fn in fns:
            splits.append(load_single_split_dir(fn, filetype=filetype))
        return splits
    else:
        split = load_single_split_dir(split_dir, filetype=filetype)
        return split


def load_additional_info(split_dir):
    additional_info_dir = join(split_dir, "additional_info")
    if not isdir(additional_info_dir):
        raise FileNotFoundError("additional_info directory doesn't exist: {}".format(additional_info_dir))

    fns = [join(additional_info_dir, f) for f in os.listdir(additional_info_dir)]
    if len(fns) == 0:
        raise FileNotFoundError("additional_info directory is empty: {}".format(additional_info_dir))

    additional_info = {}
    for fn in fns:
        # key based on specific additional info, leaving option for future additional info in different format
        if basename(fn) in ["pool_dataset_idxs", "pool_seq_positions"]:
            additional_info[basename(fn)] = load_single_split_dir(fn)
        elif basename(fn) in ["pool_muts"]:
            additional_info[basename(fn)] = load_single_split_dir(fn, content="txt")

    return additional_info


def kfold(n_samples, n_splits, rseed=None, out_dir=None, overwrite=False):
    if rseed is None:
        rseed = random.randint(1000, 9999)

    # set the random seed
    np.random.seed(rseed)

    # indices into the dataset
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)

    folds = np.array_split(idxs, n_splits)

    # choose which folds are train, val, test
    fold_splits = []
    fold_indices = list(range(n_splits))
    for i in range(n_splits):
        val = fold_indices[-(i + 2) % len(fold_indices)]
        test = fold_indices[-(i + 1) % len(fold_indices)]

        train = fold_indices.copy()
        train.remove(test)
        train.remove(val)

        fold_splits.append([train, val, test])

    # the actual train/val/test splits based on the fold indices
    splits = []
    for fs in fold_splits:
        split = {"train": np.hstack([folds[x] for x in fs[0]]),
                 "val": folds[fs[1]],
                 "test": folds[fs[2]]}
        splits.append(split)

    out_dir_split = None
    if out_dir is not None:
        out_dir_split = join(out_dir, "kfold_s{}_r{}".format(n_splits, rseed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split))
        else:
            logger.info("saving kfold split to directory {}".format(out_dir_split))

            for i, split in enumerate(splits):
                out_dir_split_rep = join(out_dir_split, basename(out_dir_split) + "_rep_{}".format(i))
                save_split(split, out_dir_split_rep)

    return splits, out_dir_split


def load_kfold_split_as_df(split_dir):
    """ load a kfold split as a dataframe """
    rep_dirs = [join(split_dir, x) for x in os.listdir(split_dir) if isdir(join(split_dir, x))]
    rep_dirs = sorted(rep_dirs, key=lambda x: int(x.split("_")[-1]))

    # load splits into a dictionary where the key is the rep num and the value is the split dict
    splits = {}
    for rd in rep_dirs:
        rep_num = int(rd.split("_")[-1])
        split = load_split_dir(rd)
        splits[f"rep_{rep_num}"] = split

    # now construct a dataframe
    # for each rep_num, we create a column in order of indices that lists train, val, or test
    # first determine number of examples in dataset
    num_examples = sum([len(v) for v in splits["rep_0"].values()])

    df = pd.DataFrame(index=range(num_examples))
    df["index"] = range(num_examples)
    for rep_num, split in splits.items():
        df[rep_num] = ""
        for set_name, idxs in split.items():
            # using .loc here is okay, even though we mean numerical indexing
            # because our pandas indexing is also set to the numerical index
            # and doing it this way avoids a pandas setting value on a copy warning
            df.loc[idxs, rep_num] = set_name

    return df


def main():
    pass


if __name__ == "__main__":
    main()
