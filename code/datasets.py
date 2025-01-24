""" custom PyTorch datasets used in METL """

from os.path import dirname, join, isdir
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor

from . import constants
from . import split_dataset as sd
from . import encode as enc


def load_standardization_params(split_dir, train_only=True):
    # if train_only True, then will only load standardization params for the training set (filename energy_X_train)
    # otherwise, will load standardization params for the full dataset (filename energy_x_all)
    standardization_params_dir = join(split_dir, "standardization_params")
    if not isdir(standardization_params_dir):
        raise FileNotFoundError("standardization_params directory doesn't exist: {}".format(standardization_params_dir))

    if train_only:
        means_fn = join(standardization_params_dir, "energy_means_train.tsv")
        stds_fn = join(standardization_params_dir, "energy_stds_train.tsv")
    else:
        means_fn = join(standardization_params_dir, "energy_means_all.tsv")
        stds_fn = join(standardization_params_dir, "energy_stds_all.tsv")

    std_params = {"means": pd.read_csv(means_fn, sep="\t", index_col="pdb_fn"),
                  "stds": pd.read_csv(stds_fn, sep="\t", index_col="pdb_fn")}

    return std_params


class DMSDataset(torch.utils.data.Dataset):
    """ Dataset for DMS data, in-memory, similar to PyTorch's TensorDataset, but support for PDB fn
        and dictionary return value """

    def __init__(self, inputs: Tensor, targets: Tensor, pdb_fn: str = None) -> None:
        # DMS datasets only support one PDB_fn for the whole dataset, so pdb_fn is a single string
        self.inputs = inputs
        self.targets = targets
        self.pdb_fn = pdb_fn

    def __getitem__(self, index):
        out_dict = {"inputs": self.inputs[index]}

        if self.targets is not None:
            out_dict["targets"] = self.targets[index]

        if self.pdb_fn is not None:
            out_dict["pdb_fns"] = self.pdb_fn

        return out_dict

    def __len__(self):
        return self.inputs.size(0)


class RosettaDatasetSQL(torch.utils.data.Dataset):
    """ Rosetta dataset from sqlite3 which can be read off-disk.
        future versions of codebase should move away from sqlite3.
        an alternative would be to use an HDF5 file to store the energies
        and a CSV file for storing the PDB fns. """

    def __init__(self,
                 db_fn: str,
                 split_dir: str,
                 set_name: str,
                 # target_names is optional to support inference-only mode where there will be no targets
                 target_names: Optional[list[str]],
                 encoding: str):

        # set fields
        self.db_fn = db_fn
        self.split_dir = split_dir
        self.target_names = target_names
        self.encoding = encoding
        self.set_name = set_name

        # the indices (into the full database) of the current set, used for converting indices in __getitem__
        self.set_idxs = sd.load_split_dir(split_dir)[set_name]

        # global PDB index
        self.pdb_index = pd.read_csv("data/rosetta_data/pdb_index.csv", index_col="pdb_fn")

        # get indices of pdb_fn, mutations, and target cols
        # needed because sql query result is a numbered array rather than a named dataframe
        col_names = self.get_col_names()
        self.pdb_col = col_names.index("pdb_fn")
        self.mutations_col = col_names.index("mutations")
        self.target_cols = None
        if target_names is not None:
            self.target_cols = [col_names.index(target_name) for target_name in target_names]

        # energy means and standard deviations used for standardizing data on-the-fly
        # note this loads the means and stds for *all* energies, but we only need them for the *target* energies
        # train_only signifies standardization params only computed on training set... should always be the case
        self.energy_means = None
        self.energy_std = None
        if split_dir is not None and target_names is not None:
            # currently split_dir is always set, but check for it in case we want to change that in the future
            # check if target_names because we don't need to load the standardization params if we're not using them
            standardization_params = self.load_standardization_params(train_only=True)
            self.energy_means = standardization_params["means"]
            self.energy_stds = standardization_params["stds"]

    def get_col_names(self):
        # create a connection to the database to load up the column names from database
        # must run a dummy query to populate column names by selecting the first rowid
        con = sqlite3.connect(self.db_fn)
        cur = con.cursor()
        cur.execute("SELECT * FROM `variant` WHERE ROWID==1")
        col_names = list(map(lambda x: x[0], cur.description))
        cur.close()
        con.close()
        return col_names

    def load_standardization_params(self, train_only=True):
        # train_only should always be used
        if train_only:
            # training set standardization params stored in split directory
            std_params = load_standardization_params(self.split_dir)
        else:
            # all data standardization params stored in ds_dir
            ds_dir = dirname(self.db_fn)
            std_params = load_standardization_params(ds_dir)
        return std_params

    def __getitem__(self, set_idx):
        # currently, have to create & destroy the database connection on each call to __getitem__
        # because it can't pickle the sqlite3 connection object. this introduces overhead.
        # future versions of this codebase should move away from sqlite3.
        self.con = sqlite3.connect(self.db_fn)
        self.cur = self.con.cursor()

        # idx argument indexes into the *set*, but the database contains variants from all sets
        # need to add 1 because the database rowid is 1-indexed
        db_idx = self.set_idxs[set_idx] + 1

        # query the database for the variant at the given index
        query = "SELECT * FROM `variant` WHERE ROWID=={}".format(db_idx)
        result = self.cur.execute(query).fetchall()[0]

        # grab info about this variant and pdb file
        variant = result[self.mutations_col]
        pdb_fn = result[self.pdb_col]
        wt_aa = self.pdb_index.loc[pdb_fn]["aa_sequence"]

        # encode the variant
        enc_variant = enc.encode(encoding=self.encoding,
                                 variants=variant,
                                 wt_aa=wt_aa,
                                 # no offset for any of the PDBs used for Rosetta
                                 wt_offset=0,
                                 # specify indexing = 1_indexed as these variants are coming from the rosetta database
                                 indexing="1_indexed")

        # get the target energies as a numpy array -- note this selects only the target_names columns.
        # when standardizing, must make sure to also select corresponding means & stds
        targets = None
        if self.target_cols is not None:
            targets = np.array([result[i] for i in self.target_cols], dtype=np.float32)
            # standardize energies using pre-computed means and standard deviations
            # if std is zero for any of the energies, then the final standardized result should be zero
            target_means = self.energy_means.loc[pdb_fn][self.target_names].to_numpy()
            target_stds = self.energy_stds.loc[pdb_fn][self.target_names].to_numpy()

            targets = np.divide((targets - target_means), target_stds, out=np.zeros_like(targets),
                                where=target_stds != 0)

        # must close *and* remove references
        self.cur.close()
        self.con.close()
        self.cur = None
        self.con = None

        out_dict = {"inputs": torch.from_numpy(enc_variant),
                    "pdb_fns": pdb_fn}

        if targets is not None:
            out_dict["targets"] = torch.from_numpy(targets)

        return out_dict

    def __len__(self):
        return len(self.set_idxs)


def pad_sequences_collate_fn(batch):
    """ a collate_fn for PyTorch dataloader that pads sequences of different lengths.
        meant for use w/ RosettaDatasetSQL, will return matching dictionary structure.
        note that this functionality is no longer for our method, but is kept here for reference.
        https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py """

    # assuming this is used with RosettaDatasetSQL, the input 'batch' should be a list of dictionaries
    # each dictionary comes from RosettaDatasetSQL and contains keys
    #   inputs: torch array w/ encoded variant (either one-hot or int_seqs)
    #   pdb_fns: the pdb fn associated with the variant
    #   targets: torch array w/ target labels (Rosetta energies)

    inputs = [d["inputs"] for d in batch]
    pdb_fns = [d["pdb_fns"] for d in batch]
    targets = [d["targets"] for d in batch]

    # save original sequence lengths and pad sequences to largest in batch
    lengths = torch.LongTensor([len(seq) for seq in inputs])
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=constants.C2I_MAPPING_2["PAD"])

    # collate pdb_fns and targets (using default collation)
    pdb_fns = torch.utils.data.default_collate(pdb_fns)
    targets = torch.utils.data.default_collate(targets)

    return {"inputs": inputs, "lengths": lengths, "pdb_fns": pdb_fns, "targets": targets}
