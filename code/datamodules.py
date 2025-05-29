""" PyTorch Lightning datamodules for DMS and Rosetta datasets """

import warnings
from argparse import ArgumentParser
from os.path import dirname, join
from typing import Optional, Union, Any

import numpy as np
import pandas as pd

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import pytorch_lightning as pl

try:
    from . import datasets
    from . import pdb_sampler
    from . import utils
    from . import constants
    from . import split_dataset as sd
    from . import encode as enc
    from .datasets import RosettaDatasetSQL
except ImportError:
    import datasets
    import pdb_sampler
    import utils
    import constants
    import split_dataset as sd
    import encode as enc
    from datasets import RosettaDatasetSQL


class DMSDataModule(pl.LightningDataModule):

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--ds_name",
                            help="name of the dms dataset defined in datasets.yml",
                            type=str, required=True)
        parser.add_argument('--pdb_fn', type=str, default="auto",
                            help="pdb file for relative_3D position encoding")
        parser.add_argument("--encoding",
                            help="which data encoding to use",
                            type=str, default="one_hot")
        parser.add_argument("--target_names",
                            help="the names of the target variables "
                                 "(currently only supports one target variable)",
                            type=str, default=None)
        parser.add_argument("--shuffle_targets",
                            help="whether to shuffle the target labels/scores, for debugging",
                            action="store_true")
        parser.add_argument("--target_roll",
                            help="how much to shift the targets relative to the variants, for debugging",
                            default=0, type=int)
        parser.add_argument("--standardize_targets",
                            help="whether to standardize targets using train set",
                            action="store_true")
        parser.add_argument("--target_offset",
                            help="an offset to add to every target value in the dataset",
                            type=float, default=0)
        parser.add_argument("--aux_input_names",
                            help="the names of the auxiliary inputs which the model can access at any layer",
                            type=str, nargs="+", default=None)
        parser.add_argument("--aux-formats",
                            help="Comma-separated list of aux input formats, e.g. 'feat1=tensor,feat2=numpy,feat3=string'."
                                 " Valid formats are 'tensor', 'numpy', and 'string'. If not specified, all numerical aux inputs"
                                 " will be converted to tensors, and all string aux inputs will be left as strings.",
                            type=str, default="")
        parser.add_argument("--aux_input_num",
                            help="number of auxiliary inputs",
                            type=int,default=1)
        parser.add_argument("--split_dir",
                            help="the directory containing the train/tune/test split",
                            type=str, default=None)
        parser.add_argument("--use_val_for_training",
                            help="whether to combine the val set with the train set for training",
                            action="store_true")
        parser.add_argument("--train_name",
                            help="name of the train set in the split dir",
                            type=str, default="train")
        parser.add_argument("--val_name",
                            help="name of the validation set in the split dir",
                            type=str, default="val")
        parser.add_argument("--test_name",
                            help="name of the test set in the split dir",
                            type=str, default="test")
        # predict mode
        parser.add_argument("--predict_mode",
                            help="what predict mode to use",
                            type=str, default="all_sets",
                            choices=["all_sets", "full_dataset", "wt"])

        parser.add_argument("--batch_size",
                            help="batch size for the data loader and optimizer",
                            type=int, default=32)

        parser.add_argument("--num_dataloader_workers",
                            help="number of workers for the data loader",
                            type=int, default=4)

        parser.add_argument("--num_classes",
                            help="number of classes if doing ordinal regression, necessary for MSE importance weighting in loss",
                            type=int,
                            default=None)
        parser.add_argument("--set_importance_weights",
                            help= """Weights to assign to ordinal classes. For MSE loss, the weight vector should have dimension equal to the number of classes. 
                                    For ranking tasks such as CORN or CORAL loss, the vector dimension should be (num_classes - 1). 
                                    If None is passed (i.e., not specified), the importance weights will be automatically calculated based on 
                                    class prevalence in the training set. 
                                    Note: To use importance weights, class labels must range from 0 to N-1, where N is the number of classes.""",
                            type=float, default=None, nargs='+')
        parser.add_argument("--use_importance_weights",
                            help="""If you wish to use importance weights, you must specify this flag.
                                    If --importance_weights is not provided, they will be automatically calculated based on 
                                    class prevalence in the training set.
                                    Note: To use importance weights, class labels must range from 0 to N-1, 
                                    where N-1 is the highest-ranking class.
                                    Must specify --num_classes if using importance weights with MSE loss.""",
                                #todo: provide an error message telling the user that you must pass in these
                                # labels for ordinal work.
                             action="store_true")

        return parser

    def __init__(self,
                 ds_name: str,
                 pdb_fn: Optional[str] = "auto",
                 encoding: str = "one_hot",
                 flatten_encoded_data: bool = False,
                 target_names: Optional[Union[str, list[str], tuple[str]]] = None,
                 split_dir: Optional[str] = None,
                 use_val_for_training: bool = False,
                 shuffle_targets: bool = False,
                 target_roll: int = 0,
                 standardize_targets: bool = False,
                 target_offset: float = 0,
                 aux_input_names: Optional[Union[str, list[str], tuple[str]]] = None,
                 aux_formats: Optional[str] = None,
                 train_name: str = "train",
                 val_name: str = "val",
                 test_name: str = "test",
                 batch_size: int = 32,
                 predict_mode: str = "all_sets",
                 num_dataloader_workers: int = 4,
                 use_importance_weights: bool = False,
                 set_importance_weights: list = None,
                 *args, **kwargs):

        super().__init__()

        # basic dataset and encoding info
        self.ds_name = ds_name
        # load dataset metadata
        self.ds_metadata = utils.load_dataset_metadata()[self.ds_name]
        self.pdb_fn = None
        self._init_pdb_fn(pdb_fn)

        # for compatability with RosettaDataset and help setting up Relative3D RPE
        self.unique_pdb_fns = None if self.pdb_fn is None else [self.pdb_fn]
        # encoding such as one_hot, int_seqs, etc
        self.encoding = encoding
        self.flatten_encoded_data = flatten_encoded_data
        # number of tokens needed in model gen code to set up the embedding layer
        self.num_tokens = constants.NUM_CHARS
        # batch size is needed for the data loader
        self.batch_size = batch_size

        # the directory containing the train/val/test split and the set names within that dir
        self.split_dir = split_dir
        self.use_val_for_training = use_val_for_training
        # load dictionary containing split indices
        # note: must use self.set_name_map or self.<set>_name when selecting a set from split_idxs
        # in this module, "set_name" refers to the module set names train, val test
        # and "user_set_name" refers to user set names given as arguments self.train_name, etc
        self.split_idxs = None
        self.has_val_set = None
        self._init_split_dir(train_name, val_name)

        # because we are using auto_test_name, need to load the split directory before
        # setting the self.test_name because the function needs to check what set names are in the split
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = self._auto_test_name(test_name)
        self.set_name_map = {"train": self.train_name, "val": self.val_name, "test": self.test_name}

        # setting for dataloaders
        self.predict_mode = predict_mode
        self.num_dataloader_workers = num_dataloader_workers

        # load the pandas dataframe for the dataset
        self.ds = utils.load_dataset(ds_name=ds_name)

        # set up and verify target_names (must be called after dataset is loaded in self.ds)
        self.target_names = None
        self.num_tasks = None
        self._init_target_names(target_names)
        self.shuffle_targets = shuffle_targets
        self.target_offset = target_offset
        # roll targets by this amount, for debugging
        self.target_roll = target_roll
        if self.target_names is not None and self.target_roll is not None and self.target_roll != 0:
            for target_name in self.target_names:
                self.ds[target_name] = np.roll(self.ds[target_name], shift=self.target_roll)

        # standardization parameters for targets, calculated only on train set if
        # a train set exists, otherwise calculated on the full dataset
        self.standardize_targets = standardize_targets
        self.target_standardize_means = None
        self.target_standardize_stds = None
        if self.standardize_targets:
            self._calc_target_standardize_params()

        # standardization parameters for when using rosetta energies as input features, calculated only on train set
        self.input_standardize_means = None
        self.input_standardize_stds = None
        if enc.is_rosetta_encoding(encoding):
            warnings.warn("detected rosetta encoding, calculating standardization parameters for input features")
            self._calc_input_standardize_params()

        # set up auxiliary inputs
        self.aux_input_names = None
        self._init_aux_input_names(aux_input_names)
        self.aux_input_formats = self.parse_aux_formats(aux_formats)

        # initialize DMSDataset that are used later in this module to load dataloaders, etc
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.full_ds = None

        # amino acid sequence length
        self.aa_seq_len = len(self.ds_metadata["wt_aa"])

        # the last couple of properties depend on getting a sample batch, so we can set the example
        # input array and data encoding lengths used to construct some models
        self.example_input_array = None
        self.aa_encoding_len = None
        self.seq_encoding_len = None
        sample_encoded_data_batch = self.get_sample_encoded_data_batch()
        sample_aux_batch = self.get_sample_aux_batch()
        if sample_encoded_data_batch is not None:
            self._init_encoding_lens(sample_encoded_data_batch)
            self._init_example_input_array(
                sample_encoded_data_batch,
                sample_aux_batch
            )



        # class imbalance training with an ordinal function
        if set_importance_weights is not None and use_importance_weights == False:
            raise ValueError(
                """--importance_weights (used for class imbalance training on ordinal data) 
            was specified, however, the flag --use_importance_weights was not specified!
            Please pass in the --use_importance_weights flag to use --importance_weights, 
            or do not specify --importance_weights at all."""
            )

        loss_func = kwargs['loss_func']
        nb_classes = kwargs['num_classes']
        if use_importance_weights:
            if set_importance_weights is None:

                # going to make an assumption the data must be labeled from 0 to N-1. (Where N is number of classes)
                count_dict = {i: 0 for i in np.arange(0,nb_classes,1)}
                unique, counts = np.unique(self._get_raw_targets("train").flatten(), return_counts=True)
                count_dict_true_counts = dict(sorted(zip(unique, counts)))
                for k, v in count_dict_true_counts.items():
                    count_dict[int(k)]=v

                for k, v in count_dict.items():
                    if v == 0:
                        count_dict[k] = 0.1

                if loss_func == 'corn':
                    # we weight via each binary output tasks (p(>k))
                    raw_importance_weights = torch.tensor(
                        [(count_dict[i] + count_dict[i + 1]) / 2 for i in np.arange(0, nb_classes - 1, 1)])
                    importance_weights_inverse = (1 / raw_importance_weights)
                    importance_weights_np = (importance_weights_inverse / importance_weights_inverse.sum()).numpy()
                    self.importance_weights = [float(w) for w in importance_weights_np]
                elif loss_func == 'mse':
                    raw_importance_weights = torch.tensor([count_dict[i] for i in np.arange(0, nb_classes, 1)])
                    importance_weights_inverse = (1 / raw_importance_weights)
                    importance_weights_np = (importance_weights_inverse / importance_weights_inverse.sum()).numpy()
                    self.importance_weights = [float(w) for w in importance_weights_np]
            else:
                # the importance weights were specified by the user
                importance_weights =  set_importance_weights
                # tensor([0.0981, 0.1324, 0.1738, 0.5957])
                # tensor([0.4048, 0.3000, 0.2286, 0.0667])
                if loss_func == 'corn':
                    assert len(importance_weights) == nb_classes - 1, \
                        'Importance length must be N-1 for ordinal based loss, where N is the number of output classes'
                elif loss_func == 'mse':
                    assert len(importance_weights) == nb_classes, \
                        'Importance length must be N for MSE, where N is the number of output classes'
                else:
                    raise ValueError('must choose a valid loss function from specified options')

                self.importance_weights = importance_weights

        else:
            # we don't want to use importance weights, the default case
            self.importance_weights = None

    def _init_pdb_fn(self, pdb_fn):
        if pdb_fn == "auto" and "pdb_fn" in self.ds_metadata:
            self.pdb_fn = self.ds_metadata["pdb_fn"]
        elif pdb_fn == "auto" and "pdb_fn" not in self.ds_metadata:
            warnings.warn("'pdb_fn' set to 'auto' but no pdb_fn found in "
                          "dataset metadata. setting pdb_fn to None")
            self.pdb_fn = None
        else:
            self.pdb_fn = pdb_fn

    def _init_example_input_array(
            self,
            sample_encoded_data_batch: np.ndarray,
            sample_aux_batch: Optional[dict[str, Any]]) -> None:

        # example input array helps with sanity checking
        # and printing full model summaries
        example_input_array = {
            "x": torch.from_numpy(sample_encoded_data_batch),
            "pdb_fn": self.pdb_fn
        }

        if sample_aux_batch is not None:
            # group aux inputs into their own dictionary
            example_input_array["aux"] = sample_aux_batch

        self.example_input_array = example_input_array

    def get_sample_encoded_data_batch(self) -> np.ndarray:
        """ compute a sample batch of encoded data used for example input array,
            sequence encoding lengths, etc """

        # encode a full batch of variants to get the encoding
        # length (the last dim) and an example_input_array
        variants = self.ds.iloc[0:self.batch_size]["variant"].tolist()
        return self.get_encoded_variants(variants)

    def get_sample_aux_batch(self) -> Optional[dict[str, Any]]:
        """ compute a sample batch of auxiliary data used for example input array """
        batch_idxs = np.arange(self.batch_size)
        return self._get_aux_inputs_for_idxs(batch_idxs)

    def _init_encoding_lens(self, sample_encoded_data_batch: np.ndarray) -> None:
        if sample_encoded_data_batch is None:
            # encoding length stuff is extra, only used for METL models, so give option not to compute it
            warnings.warn("Unable to compute aa_encoding_len and seq_encoding_len because sample_batch is None")
            return
        if len(sample_encoded_data_batch.shape) not in [2, 3]:
            raise ValueError("temp_enc_variant has an unknown shape: {}".format(sample_encoded_data_batch.shape))
        elif len(sample_encoded_data_batch.shape) == 2:
            # if 2 dimensions, then it's a seq-level encoding (batch_size, encoded_seq)
            self.aa_encoding_len = 0
            self.seq_encoding_len = sample_encoded_data_batch.shape[-1]
        elif len(sample_encoded_data_batch.shape) == 3:
            # if 3 dimensions, then it's an amino-acid level encoding (batch_size, aa_seq_len, enc_len)
            self.aa_encoding_len = sample_encoded_data_batch.shape[-1]
            self.seq_encoding_len = self.aa_seq_len * self.aa_encoding_len
            # an additional validation check just in case
            if self.aa_seq_len != sample_encoded_data_batch.shape[-2]:
                raise ValueError("expected aa_seq_len to be {}, but temp_enc_variant is {}".format(
                    self.aa_seq_len, sample_encoded_data_batch.shape[-2]))

    def _auto_test_name(self, test_name):
        if test_name != "auto":
            return test_name

        if self.split_idxs is None:
            raise ValueError("unable to determine test set name because no split directory provided")

        if "test" in self.split_idxs:
            # prioritize "test" rather than "stest"
            test_name = "test"
        elif "stest" in self.split_idxs:
            test_name = "stest"
        else:
            raise ValueError("unable to determine test set name because neither 'test' nor 'stest' are in split")

        return test_name

    def _init_split_dir(self, train_name, val_name):
        if self.split_dir is None:
            warnings.warn("Split directory is None for DMSDataModule")
        else:
            self.split_idxs = sd.load_split_dir(self.split_dir)
            if self.use_val_for_training:
                # combine the train and val set into the test set, and delete the val set
                self.split_idxs[train_name] = np.concatenate((self.split_idxs[train_name], self.split_idxs[val_name]))
                del self.split_idxs[val_name]
            self.has_val_set = val_name in self.split_idxs

    def _init_target_names(self, target_names):
        if target_names is None:
            return

        if not isinstance(target_names, list) and not isinstance(target_names, tuple):
            target_names = [target_names]

        # verify all the target names are in the dataset
        for tn in target_names:
            if tn not in self.ds:
                raise ValueError("target not found in dataset: {}".format(tn))

        self.target_names = target_names
        self.num_tasks = len(self.target_names)

    def _calc_target_standardize_params(self) -> None:
        """ calculate the means and standard deviations of all energy terms for the train set """
        # if there is no split... standardize using the full dataset, but throw a warning just in case
        if self.split_idxs is None:
            warnings.warn("Computing target standardization params using full dataset because there is no train split")
            target_vals = self._get_raw_targets(None)
        else:
            target_vals = self._get_raw_targets("train")

        # calculate mean and standard deviation
        self.target_standardize_means = np.nanmean(target_vals, axis=0)
        # ddof=0 to match sklearn's StandardScaler
        self.target_standardize_stds = np.nanstd(target_vals, axis=0, ddof=0)

    def _calc_input_standardize_params(self):
        """ calculate the means and standard deviations of all energy terms for the train set """
        standardization_set = "train"
        if self.split_idxs is None:
            # if there is no split... standardize using the full dataset, but throw a warning just in case
            standardization_set = None
            warnings.warn("Computing input standardization params using full dataset because there is no train split")

        # note: we are using concat=False so enc_data will be a list of arrays (one for each encoding)
        # we do this because we only need to standardize Rosetta encodings and this helps loop through the encodings
        enc_data = self._get_raw_encoded_data(standardization_set, concat=False)
        if not isinstance(enc_data, list):
            enc_data = [enc_data]

        # note: this assumes the data will be flattened, so the standardization parameters are flattened as well
        # we only need to standardize the rosetta-based encodings...
        standardize_means = []
        standardize_stds = []
        for e, ed in zip(self.encoding.split("+"), enc_data):
            if enc.is_rosetta_encoding(e):
                # calculate mean and standard deviation
                standardize_means.append(np.nanmean(ed, axis=0))
                # ddof=0 to match sklearn's StandardScaler
                standardize_stds.append(np.nanstd(ed, axis=0, ddof=0))
            else:
                # if not a rosetta encoding, then just use 0 and 1 for mean and standard deviation
                # note we need to use the flattened shape of this encoding because
                # it might be a residue-level encoding like one-hot
                # we assume that the encoding will eventually be flattened (when we use concat=True)
                # so using the flattened shape will match that
                if len(ed.shape) == 1:
                    # if 1D, then it's just a single value for each sample, thus flattened encoding length is 1
                    flattened_enc_shape = 1
                elif len(ed.shape) >= 2:
                    # if 2D or more, calculate the flattened encoding length by multiplying dimensions after dim 0
                    flattened_enc_shape = np.prod(ed.shape[1:])
                else:
                    raise ValueError("unknown shape for encoding: {}".format(ed.shape))

                standardize_means.append(np.zeros(flattened_enc_shape))
                standardize_stds.append(np.ones(flattened_enc_shape))

        # now concatenate the means and standard deviations for each encoding
        self.input_standardize_means = np.concatenate(standardize_means)
        self.input_standardize_stds = np.concatenate(standardize_stds)

    def _init_aux_input_names(
            self,
            aux_input_names: Optional[Union[str, list[str], tuple[str]]]) -> None:

        if aux_input_names is None:
            return

        # if the auxiliary input names are a string or tuple, convert to a list
        if isinstance(aux_input_names, str):
            aux_input_names = [aux_input_names]
        elif isinstance(aux_input_names, tuple):
            aux_input_names = list(aux_input_names)

        # verify all the auxiliary input names are in the dataset
        for ain in aux_input_names:
            if ain not in self.ds:
                raise ValueError("auxiliary input not found in dataset: {}".format(ain))

        self.aux_input_names = aux_input_names

    @staticmethod
    def parse_aux_formats(s: str) -> dict[str, str]:
        if not s:
            return {}
        out = {}
        for item in s.split(","):
            key, val = item.split("=")
            val = val.strip().lower()
            if val not in ("tensor", "numpy", "string"):
                raise ValueError(f"invalid format '{val}' for aux input '{key.strip()}'")
            out[key.strip()] = val
        return out

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        pass

    def get_variants(self, set_name: Optional[str]):
        if set_name is None:
            variants = self.ds["variant"].tolist()
        else:
            idxs = self.get_split_idxs(set_name)
            variants = self.ds.iloc[idxs]["variant"].tolist()
        return variants

    @staticmethod
    def _standardize(data: np.ndarray,
                     means: np.ndarray,
                     stds: np.ndarray) -> np.ndarray:

        if means is None or stds is None:
            raise ValueError("need to standardize, but standardize params are None")

        standardized_data = np.divide((data - means), stds, out=np.zeros_like(data),
                                      where=stds != 0)
        return standardized_data

    def _standardize_inputs(self, data: np.ndarray) -> np.ndarray:
        """ perform standardization of rosetta energies using the given means and standard deviations """
        if self.input_standardize_means is None or self.input_standardize_stds is None:
            raise ValueError("need to standardize rosetta encoded data, but standardize params are None")
        energies = self._standardize(data, self.input_standardize_means, self.input_standardize_stds)
        return energies

    def _standardize_targets(self, data):
        if self.target_standardize_means is None or self.target_standardize_stds is None:
            raise ValueError("need to standardize targets, but standardize params are None")
        targets = self._standardize(data, self.target_standardize_means, self.target_standardize_stds)
        return targets

    def _get_raw_encoded_variants(
            self,
            variants: list[str],
            concat: bool = True) -> np.ndarray:

        enc_data = enc.encode(
            encoding=self.encoding,
            variants=variants,
            ds_name=self.ds_name,
            concat=concat
        )
        return enc_data

    def get_encoded_variants(self, variants: list[str]) -> np.ndarray:
        enc_data = self._get_raw_encoded_variants(variants, concat=True)

        # standardize if using rosetta encoding
        if enc.is_rosetta_encoding(self.encoding):
            enc_data = self._standardize_inputs(enc_data)

        if self.flatten_encoded_data:
            enc_data = enc_data.reshape(enc_data.shape[0], -1)

        return enc_data

    def _get_raw_encoded_data(self, set_name: Optional[str], concat: bool = True):
        variants = self.get_variants(set_name)
        return self._get_raw_encoded_variants(variants, concat=concat)

    def get_encoded_data(self, set_name: Optional[str]) -> np.ndarray:
        variants = self.get_variants(set_name)
        return self.get_encoded_variants(variants)

    def _get_raw_targets(self, set_name: Optional[str]) -> Optional[np.ndarray]:
        if self.target_names is None:
            return None

        if set_name is None:
            targets = self.ds[self.target_names].to_numpy().astype(np.float32)
        else:
            idxs = self.get_split_idxs(set_name)
            targets = self.ds.iloc[idxs][self.target_names].to_numpy().astype(np.float32)

        return targets

    def get_targets(
            self,
            set_name: Optional[str],
            squeeze: bool = False) -> Optional[np.ndarray]:

        targets = self._get_raw_targets(set_name)
        if targets is None:
            return None

        # standardize FIRST, then add offset
        if self.standardize_targets:
            targets = self._standardize_targets(targets)

        # add target offset for testing effect of target magnitude on training...
        targets += self.target_offset

        if self.shuffle_targets:
            np.random.shuffle(targets)

        if squeeze:
            targets = targets.squeeze()

        return targets

    def _get_aux_inputs_for_idxs(self, idxs: np.ndarray) -> Optional[dict[str, Any]]:
        if self.aux_input_names is None:
            return None

        aux_inputs = {}
        for ain in self.aux_input_names:
            col = self.ds.iloc[idxs][ain]

            if ain in self.aux_input_formats:
                # if a format is specified for this auxiliary input, use it
                fmt = self.aux_input_formats[ain]
            elif pd.api.types.is_numeric_dtype(col):
                # otherwise default to pytorch tensor if column is numeric
                fmt = "tensor"
            else:
                fmt = "string"

            if fmt == "tensor":
                if not pd.api.types.is_numeric_dtype(col):
                    raise TypeError(
                        f"Cannot convert non-numeric column '{ain}' to tensor.")
                aux_inputs[ain] = torch.from_numpy(col.to_numpy())
            elif fmt == "numpy":
                if not pd.api.types.is_numeric_dtype(col):
                    raise TypeError(
                        f"Cannot convert non-numeric column '{ain}' to numpy array.")
                aux_inputs[ain] = col.to_numpy()
            elif fmt == "string":
                aux_inputs[ain] = col.tolist()
            else:
                raise ValueError(f"Invalid aux input format '{fmt}' for column '{ain}'")

        return aux_inputs

    def get_aux_inputs(self, set_name: Optional[str]) -> Optional[dict[str, Any]]:
        if set_name is None:
            # get indexes for the entire dataset
            idxs = np.arange(len(self.ds))
        else:
            # get indexes for the given set name
            idxs = self.get_split_idxs(set_name)

        return self._get_aux_inputs_for_idxs(idxs)

    def has_set(self, set_name: str, user_set_name: bool = False):
        if self.split_idxs is None:
            return False
        if user_set_name:
            return set_name in self.split_idxs
        else:
            return self.set_name_map[set_name] in self.split_idxs

    def get_split_idxs(self, set_name: Optional[str] = None, user_set_name: bool = False):
        """ get the indices for the given set name, or all indices if set_name is None
            if user_set_name is True, then the set_name is assumed to be the user-set name, not the internal name """

        if set_name is None:
            return self.split_idxs
        else:
            if user_set_name:
                # the provided set_name is a user-defined set name given in the split file
                # self.split_idxs is keyed by user set names, so we can access it directly
                return self.split_idxs[set_name]
            else:
                # the provided set_name is an internal set name, either "train", "val", or "test"
                # self.set_name_map maps internal set names to user-defined set names, so we can index self.split_idxs
                return self.split_idxs[self.set_name_map[set_name]]

    def get_ds(self, set_name: Optional[str]):
        # special handling for wild-type only variant
        if set_name == "_wt":
            targets = None
            enc_data = self.get_encoded_variants(["_wt"])
            # todo: how should we handle the auxiliary inputs for the wild-type variant?
            aux_inputs = None
        else:
            targets = self.get_targets(set_name)
            enc_data = self.get_encoded_data(set_name)
            aux_inputs = self.get_aux_inputs(set_name)

        torch_ds = datasets.DMSDataset(inputs=torch.from_numpy(enc_data),
                                       targets=None if targets is None else torch.from_numpy(targets),
                                       pdb_fn=self.pdb_fn,
                                       aux_inputs=aux_inputs)
        return torch_ds

    def setup(self, stage=None):
        # verify split_dir and target_names were provided if this datamodule is used for fitting or testing
        # predicting doesn't require a split_dir because prediction can be done on full dataset
        if stage == "fit" or stage == "test":
            if self.split_dir is None:
                raise ValueError("datamodule is being set up for: {}, but split_dir is None".format(stage))
            if self.target_names is None:
                raise ValueError("datamodule is being set up for: {}, but target_names is None".format(stage))

        if stage == "fit" or stage is None:
            # if stage is None, but a split dir is provided, load the train_ds and val_ds
            if self.split_dir is not None and self.target_names is not None:
                self.train_ds = self.get_ds("train")
                if self.has_val_set:
                    self.val_ds = self.get_ds("val")

        if stage == "test" or stage is None:
            # if stage is None, but a split dir is provided, load the test_ds
            if self.split_dir is not None and self.target_names is not None:
                self.test_ds = self.get_ds("test")

        if stage == "predict" or stage is None:
            if self.predict_mode == "full_dataset":
                self.full_ds = self.get_ds(None)
            elif self.predict_mode == "wt":
                self.wt_ds = self.get_ds("_wt")
            elif self.predict_mode == "all_sets":
                # just in case, make sure all the split datasets have been set up
                # this should already be the case if the model was just trained with this datamodule
                if self.train_ds is None:
                    self.train_ds = self.get_ds("train")
                if self.has_val_set and self.val_ds is None:
                    self.val_ds = self.get_ds("val")
                if self.test_ds is None:
                    self.test_ds = self.get_ds("test")

    def _get_dataloader(self, ds):
        """ helper function for loading train, val, and test dataloaders """
        if ds is None:
            # return None for the dataloader if the underlying dataset is None
            # handles the case for when there is no validation set and val dataloader should be None
            return None
        else:
            return data_utils.DataLoader(ds,
                                         batch_size=self.batch_size,
                                         num_workers=self.num_dataloader_workers,
                                         persistent_workers=True if self.num_dataloader_workers > 0 else False)

    def train_dataloader(self):
        return self._get_dataloader(self.train_ds)

    def val_dataloader(self):
        return self._get_dataloader(self.val_ds)

    def test_dataloader(self):
        return self._get_dataloader(self.test_ds)

    def predict_dataloader(self):
        if self.predict_mode == "all_sets":
            return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]
        elif self.predict_mode == "train_set":
            return self.train_dataloader()
        elif self.predict_mode == "full_dataset":
            return self._get_dataloader(self.full_ds)
        elif self.predict_mode == "wt":
            return self._get_dataloader(self.wt_ds)
        else:
            raise ValueError("unknown predict mode: {}".format(self.predict_mode))


class RosettaDataModule(pl.LightningDataModule):

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--ds_fn",
                            help="filename of the csv/hdf5 dataset",
                            type=str, required=True)

        parser.add_argument("--encoding",
                            help="which data encoding to use.",
                            type=str, default="int_seqs")

        parser.add_argument("--split_dir",
                            help="the directory containing the train/tune/test split",
                            type=str, required=True)
        parser.add_argument("--train_name",
                            help="name of the train set in the split dir",
                            type=str, default="train")
        parser.add_argument("--val_name",
                            help="name of the validation set in the split dir",
                            type=str, default="val")
        parser.add_argument("--test_name",
                            help="name of the test set in the split dir",
                            type=str, default="test")

        parser.add_argument("--target_group",
                            help="which group of energies to use as targets. "
                                 "if set, overrides both target_names and target_names_exclude",
                            type=str, default="standard",
                            choices=["standard-all", "standard", "standard-docking", "docking"])
        parser.add_argument("--target_names",
                            help="names of rosetta energies to use as targets (overrides exclude)",
                            type=str, nargs="+", default=None)
        parser.add_argument("--target_names_exclude",
                            help="include all STANDARD (non-docking) energies, except these",
                            type=str, nargs="*", default=None)

        parser.add_argument("--batch_size",
                            help="batch size for the data loader and optimizer",
                            type=int, default=32)

        return parser

    def __init__(self,
                 ds_fn: str,
                 encoding: str,
                 split_dir: str,
                 train_name: str = "train",
                 val_name: str = "val",
                 test_name: str = "test",
                 batch_size: int = 32,
                 # target tasks
                 target_group: Optional[str] = None,
                 target_names: Optional[Union[list[str], tuple[str]]] = None,
                 target_names_exclude: Union[list[str], tuple[str]] = (),
                 # whether to use the distributed sampler for the train and val dataloaders
                 # the test dataloader does NOT use the distributed sampler, regardless of this setting
                 enable_distributed_sampler: bool = False,
                 # these can probably be combined into a single arg: model_supports_multiple_pdbs_in_batch
                 # whether to use the PDB sampler **if the number of unique PDBs is > 1**
                 # supports the global CNN model, which doesn't need the PDB sampler
                 enable_pdb_sampler: bool = True,
                 # whether to use datasets.pad_sequences_collate_fn for dataloaders (for when enable_pdb_sampler=False)
                 # supports the global CNN model, which supports multiple PDBs in a single batch
                 use_padding_collate_fn: bool = False,
                 *args, **kwargs):

        super().__init__()

        # database fn
        self.ds_fn = ds_fn

        # target/task names and number of tasks
        self.target_names = utils.get_rosetta_energy_targets(
            target_group=target_group,
            target_names=target_names,
            target_names_exclude=target_names_exclude
        )
        self.num_tasks = len(self.target_names)

        # the dir containing the train/val/test split and the set names within that dir
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.split_dir = split_dir
        self.split = None
        self._init_split()

        # batch size is needed for the data loader
        self.batch_size = batch_size

        # the data encoding
        self.encoding = encoding

        # number of tokens needed in model gen code to set up the embedding layer
        self.num_tokens = constants.NUM_CHARS

        # load PDB fn for each example, used for PDBSampler dataloader
        pdb_fns_path = join(dirname(self.ds_fn), "pdb_fns.txt")
        self.pdb_fns = pd.read_csv(pdb_fns_path, header=None).iloc[:, 0].to_numpy()
        # split PDB fns for train/val/test sets (helps with batch sampler)
        self.pdb_fns_split = {set_name: self.pdb_fns[self.split[set_name]].tolist()
                              for set_name in [self.train_name, self.val_name, self.test_name]}
        # the unique PDB fns used in the dataset (splits)
        # todo: sort this so it is always in the same order
        #   is this the reason the buffers weren't syncing across processes w/ DDP?
        self.unique_pdb_fns = list(set().union(*[set(v) for v in self.pdb_fns_split.values()]))

        # for determining what kind of samplers to use
        self.enable_distributed_sampler = enable_distributed_sampler
        self.enable_pdb_sampler = enable_pdb_sampler
        self.use_pdb_sampler = False
        if enable_pdb_sampler and len(self.unique_pdb_fns) > 1:
            self.use_pdb_sampler = True
        self.use_padding_collate_fn = use_padding_collate_fn

        # error checking for drop_last and distributed sampler when batch size is larger than dataset size
        if self.enable_distributed_sampler:
            for set_name in [self.train_name, self.val_name, self.test_name]:
                if len(self.split[set_name]) < self.batch_size:
                    raise ValueError("Batch_size {} is larger than the number of examples in set '{}'. "
                                     "This is incompatible with enable_distributed_sampler, "
                                     "which requires drop_last=True".format(self.batch_size, set_name))

        # length of the longest sequence in dataset
        self.pdb_index = pd.read_csv("data/rosetta_data/pdb_index.csv", index_col="pdb_fn")

        # aa_seq_len is the length of the longest sequence in the dataset
        self.aa_seq_len = max(self.pdb_index.loc[self.unique_pdb_fns]["seq_len"])

        # full_seq_len factors in sequence length increase due to CLS token and potential future padding
        self.full_seq_len = self.aa_seq_len

        # aa_encoding_len and seq_encoding_len
        if self.encoding == "int_seqs":
            # int seqs use a single integer to represent each possible amino acid token
            self.aa_encoding_len = 1
            self.seq_encoding_len = self.full_seq_len * self.aa_encoding_len
        elif self.encoding == "one_hot":
            # one_hot doesn't support cls tokens, but can still use self.num_tokens here
            self.aa_encoding_len = self.num_tokens
            self.seq_encoding_len = self.full_seq_len * self.aa_encoding_len
        else:
            raise ValueError("unsupported encoding {}".format(self.encoding))

        self.example_input_array = self.get_example_input_array()

    def _init_split(self):
        self.split = sd.load_split_dir(self.split_dir)

        # check that the split dir contains the train/val/test sets
        for set_name in [self.train_name, self.val_name, self.test_name]:
            if set_name not in self.split:
                raise ValueError("split dir '{}' does not contain set '{}'".format(self.split_dir, set_name))

    def get_example_input_array(self):
        """ set up the example input array """

        # use the first pdb file in self.unique_pdb_fns as the example PDB file
        # we need to do this because we are assuming one PDB per batch
        # so, we need to choose a PDB file for the example batch
        example_pdb_fn = self.unique_pdb_fns[0]
        example_aa_seq_len = self.pdb_index.loc[example_pdb_fn]["seq_len"]
        example_full_seq_len = example_aa_seq_len

        # log some info about the example input array
        print("Using example_input_array with pdb_fn='{}' and aa_seq_len={}".format(example_pdb_fn, example_aa_seq_len))

        if self.encoding == "int_seqs":
            arr = torch.randint(low=0, high=self.num_tokens, size=(self.batch_size, example_full_seq_len))
        elif self.encoding == "one_hot":
            example_indices = torch.randint(low=0, high=self.num_tokens, size=(self.batch_size, example_full_seq_len))
            arr = F.one_hot(example_indices, self.num_tokens).float()
        else:
            raise ValueError("unsupported encoding for example_input_array: {}".format(self.encoding))

        # use a dict as the example input array because we want to pass in the PDB file.
        # this must be compatible with the task's forward() method... it is.
        # but note that the task's forward() method accepts different args than outputted by
        # RosettaDatasetSQL. the Task processes the dict from RosettaDatasetSQL in _shared_step
        # but if the forward method ever changes to take dicts, we need to account for it here.
        return {"x": arr, "pdb_fn": example_pdb_fn}

    def prepare_data(self):
        # prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # use this method to do things that might write to disk or that need to be done only from a single process
        # in distributed settings.
        pass

    def get_ds(self, set_name: str) -> torch.utils.data.Dataset:
        torch_ds = RosettaDatasetSQL(
            db_fn=self.ds_fn,
            split_dir=self.split_dir,
            set_name=set_name,
            target_names=self.target_names,
            encoding=self.encoding,
        )
        return torch_ds

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.get_ds(self.train_name)
            self.val_ds = self.get_ds(self.val_name)

        if stage == 'test' or stage is None:
            self.test_ds = self.get_ds(self.test_name)

    def get_dataloader(self,
                       set_name: str,
                       ds: torch.utils.data.Dataset,
                       shuffle: bool,
                       use_distributed_sampler: bool = False,
                       num_workers: int = 4) -> DataLoader:

        # use persistent workers if num_workers > 0
        persistent_workers = True if num_workers > 0 else False

        # whether to drop the last batch if it is smaller than the batch size
        # automatically enabled when using a distributed sampler
        # otherwise, it is disabled
        drop_last = True if use_distributed_sampler else False

        if use_distributed_sampler and not self.use_pdb_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle, drop_last=drop_last)
            return DataLoader(ds,
                              batch_size=self.batch_size,
                              persistent_workers=persistent_workers,
                              sampler=sampler,
                              num_workers=num_workers,
                              collate_fn=datasets.pad_sequences_collate_fn if self.use_padding_collate_fn else None)

        elif use_distributed_sampler and self.use_pdb_sampler:
            pdb_fns = self.pdb_fns_split[set_name]
            sampler = pdb_sampler.PDBSamplerDistributed(pdb_fns,
                                                        batch_size=self.batch_size,
                                                        shuffle=shuffle,
                                                        drop_last=drop_last)
            return DataLoader(ds,
                              batch_sampler=sampler,
                              persistent_workers=persistent_workers,
                              num_workers=num_workers)

        elif not use_distributed_sampler and not self.use_pdb_sampler:
            return DataLoader(ds,
                              batch_size=self.batch_size,
                              persistent_workers=persistent_workers,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              num_workers=num_workers,
                              collate_fn=datasets.pad_sequences_collate_fn if self.use_padding_collate_fn else None)

        elif not use_distributed_sampler and self.use_pdb_sampler:
            pdb_fns = self.pdb_fns_split[set_name]
            sampler = pdb_sampler.PDBSampler(pdb_fns,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last)
            return DataLoader(ds,
                              batch_sampler=sampler,
                              persistent_workers=persistent_workers,
                              num_workers=num_workers)

    def train_dataloader(self):
        return self.get_dataloader(self.train_name, self.train_ds, shuffle=True,
                                   use_distributed_sampler=self.enable_distributed_sampler, num_workers=4)

    def val_dataloader(self):
        return self.get_dataloader(self.val_name, self.val_ds, shuffle=False,
                                   use_distributed_sampler=self.enable_distributed_sampler, num_workers=4)

    def test_dataloader(self):
        return self.get_dataloader(self.test_name, self.test_ds, shuffle=False,
                                   use_distributed_sampler=False, num_workers=4)


class BasicRosettaDataModule(pl.LightningDataModule):
    """ a very basic datamodule for Rosetta datasets made for inference / prediction
        this should ultimately be merged with the main RosettaDataModule, but
        keeping it separate made for easier development for current purposes """

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--ds_fn",
            help="filename of the csv/hdf5 dataset",
            type=str,
            required=True
        )
        parser.add_argument(
            "--encoding",
            help="which data encoding to use",
            type=str,
            default="int_seqs"
        )
        parser.add_argument(
            "--split_dir",
            help="the directory containing the train/tune/test split.",
            type=str,
            default=None
        )
        parser.add_argument(
            "--batch_size",
            help="batch size for the data loader",
            type=int, default=32
        )
        parser.add_argument(
            "--predict_mode",
            help="prediction mode",
            type=str,
            default="full_dataset"
        )

        return parser

    def __init__(self,
                 ds_fn: str,
                 split_dir: str,
                 predict_mode: str,
                 batch_size: int = 32,
                 encoding: str = "int_seqs",
                 *args, **kwargs):

        super().__init__()

        self.ds_fn = ds_fn
        self.split_dir = split_dir
        self.split = None if split_dir is None else sd.load_split_dir(split_dir)
        self.predict_mode = predict_mode
        self._validate_predict_mode()
        self.batch_size = batch_size
        self.encoding = encoding

        # initialize list of pdb files present in the dataset (based on predict mode)
        self._init_pdb_fns()

        # load the pdb index
        self.pdb_index = pd.read_csv(
            "data/rosetta_data/pdb_index.csv",
            index_col="pdb_fn"
        )

        # automatically determine whether to use the PDB sampler
        # must be enabled if multiple unique PDBs present
        self.use_pdb_sampler = False
        if len(set(self.pdb_fns)) > 1:
            self.use_pdb_sampler = True

        self.example_input_array = self.get_example_input_array()

    def get_example_input_array(self):
        example_pdb_fn = self.pdb_fns[0]
        example_aa_seq_len = self.pdb_index.loc[example_pdb_fn]["seq_len"]
        example_full_seq_len = example_aa_seq_len

        # log some info about the example input array
        print(f"Using example_input_array with pdb_fn='{example_pdb_fn}' "
              f"and aa_seq_len={example_aa_seq_len}")

        if self.encoding == "int_seqs":
            arr = torch.randint(
                low=0,
                high=constants.NUM_CHARS,
                size=(self.batch_size, example_full_seq_len)
            )
        elif self.encoding == "one_hot":
            example_inds = torch.randint(
                low=0,
                high=constants.NUM_CHARS,
                size=(self.batch_size, example_full_seq_len)
            )
            arr = F.one_hot(example_inds, constants.NUM_CHARS).float()
        else:
            raise ValueError(f"unsupported enc for example_input_array: {self.encoding}")
        return {"x": arr, "pdb_fn": example_pdb_fn}

    def _validate_predict_mode(self):
        if self.split is None:
            valid_predict_modes = ["full_dataset"]
        else:
            valid_predict_modes = list(self.split.keys()) + ["full_dataset"]

        if self.predict_mode not in valid_predict_modes:
            raise ValueError(f"valid predict modes are: {valid_predict_modes}")

    def _init_pdb_fns(self):
        pdb_fns_path = join(dirname(self.ds_fn), "pdb_fns.txt")
        if self.predict_mode == "full_dataset":
            self.pdb_fns = pd.read_csv(pdb_fns_path, header=None).iloc[:, 0].to_numpy()
        else:
            # predict_mode is a set_name
            all_pdb_fns = pd.read_csv(pdb_fns_path, header=None).iloc[:, 0].to_numpy()
            self.pdb_fns = all_pdb_fns[self.split[self.predict_mode]].tolist()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_ds(self, set_name: str) -> torch.utils.data.Dataset:
        torch_ds = RosettaDatasetSQL(
            db_fn=self.ds_fn,
            split_dir=self.split_dir,
            set_name=set_name,
            target_names=None,
            encoding=self.encoding,
        )
        return torch_ds

    def get_dataloader(
            self,
            ds: torch.utils.data.Dataset,
            num_workers: int = 4,
            shuffle: bool = False):
        persistent_workers = True if num_workers > 0 else False
        if self.use_pdb_sampler:
            sampler = pdb_sampler.PDBSampler(self.pdb_fns,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle,
                                             drop_last=False)
            return DataLoader(ds,
                              batch_sampler=sampler,
                              persistent_workers=persistent_workers,
                              num_workers=num_workers)
        else:
            return DataLoader(ds,
                              batch_size=self.batch_size,
                              persistent_workers=persistent_workers,
                              shuffle=shuffle,
                              drop_last=False,
                              num_workers=num_workers)

    def train_dataloader(self):
        raise NotImplementedError("this datamodule is only for prediction")

    def val_dataloader(self):
        raise NotImplementedError("this datamodule is only for prediction")

    def test_dataloader(self):
        raise NotImplementedError("this datamodule is only for prediction")

    def predict_dataloader(self):
        ds = self.get_ds(self.predict_mode)
        return self.get_dataloader(ds, num_workers=4, shuffle=False)
