""" Convert lightning checkpoint to pure pytorch.
    Lightning checkpoints are compatible with pure pytorch, but they may contain additional items
    that are not needed for inference. so, this script loads the checkpoint and saves a smaller
    checkpoint with just the model weights and hyperparameters. """

import argparse
import os
import warnings
from os.path import dirname, join

import pytorch_lightning
import torch
import torch.nn as nn
import torchinfo

try:
    from . import models
    from . import utils
    from . import encode as enc
except ImportError:
    import models
    import utils
    import encode as enc


def convert_checkpoint(ckpt_dict):
    state_dict = ckpt_dict["state_dict"]
    hparams = ckpt_dict["hyper_parameters"]

    # the hparams contains all the PDB files used during training
    # with these shared models, PDBs are instead processed on the fly
    # so remove the PDB list from the hparams
    hparams["pdb_fns"] = []

    # update keys by dropping the outer "model." from the RosettaTask
    for key in list(state_dict):
        state_dict[key[len("model."):]] = state_dict.pop(key)

    # replace the pytorch_lightning.utilities.parsing.AttributeDict with a regular dict
    # this is necessary for a couple old checkpoints because they were saved with AttributeDicts
    # todo: it has been fixed in the latest version, no need to do this anymore once we get into production
    for k, v in hparams.items():
        if isinstance(v, pytorch_lightning.utilities.parsing.AttributeDict):
            print("Converting pytorch_lightning.utilities.parsing.AttributeDict to regular dict")
            hparams[k] = dict(v)

    new_ckpt_dict = {"state_dict": state_dict, "hyper_parameters": hparams}

    return new_ckpt_dict


def get_output_dir(args):
    # save just the model weights and hyperparameters to a smaller checkpoint
    output_dir = args.output_dir
    if args.output_dir == "auto":
        # if the output directory is auto, save the checkpoint in the same directory as the input
        # checkpoint, but with a different name
        output_dir = dirname(args.ckpt_path)
    return output_dir


def get_encoded_seqs(hparams):
    """ note: this only supports target models at the moment """

    ds_name = hparams["ds_name"]
    datasets = utils.load_dataset_metadata()
    pdb_fn = datasets[ds_name]["pdb_fn"]
    ds = utils.load_dataset(ds_name=ds_name)
    variants = ds["variant"].tolist()
    encoding = hparams["encoding"]

    encoded_seqs = enc.encode(encoding=encoding, variants=variants, ds_name=ds_name)
    return encoded_seqs, pdb_fn


def test_converted_checkpoint(pt_checkpoint_fn):
    # test run a variant through the converted checkpoint
    # use the relevant model-building code from the RosettaTask & hparams from checkpoint

    # load the checkpoint
    ckpt_dict = torch.load(pt_checkpoint_fn)
    state_dict = ckpt_dict["state_dict"]
    hparams = ckpt_dict["hyper_parameters"]

    # construct the model from the hparams
    model = models.Model[hparams["model_name"]].cls(**hparams)

    # print the model summary
    torchinfo.summary(model, depth=5, verbose=1, row_settings=["var_names"])

    # load the model parameters from the checkpoint
    model.load_state_dict(state_dict)

    # get encoded sequences
    encoded_seqs, pdb_fn = get_encoded_seqs(hparams)

    # set model to eval mode
    model.eval()
    # no need to compute gradients for inference
    with torch.no_grad():
        # predictions = model(torch.tensor(encoded_seqs[:10]), pdb_fn=pdb_fn)
        predictions = model(torch.tensor(encoded_seqs[10:20]), pdb_fn=pdb_fn)

    print(predictions)


def main(args):

    print(f"Processing checkpoint: {args.ckpt_path}")

    # load the lightning checkpoint with pure pytorch
    lightning_checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    pt_checkpoint = convert_checkpoint(lightning_checkpoint)

    # create output directory
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # get the UUID from the checkpoint hyperparameters (to name the checkpoint)
    uuid = pt_checkpoint["hyper_parameters"]["uuid"]
    if uuid is None:
        warnings.warn("Checkpoint does not have a UUID, using 'converted.pt' as the filename instead.")
        uuid = "converted"

    output_fn = "{}.pt".format(join(output_dir, uuid))

    # save the checkpoint with pure pytorch
    print("Saving converted checkpoint to:", output_fn)
    torch.save(pt_checkpoint, output_fn)

    # # test the converted checkpoint
    # print("Testing converted checkpoint")
    # test_converted_checkpoint(output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("--ckpt_path", required=True, type=str,
                        help="Path to the checkpoint to convert to pure pytorch.")

    parser.add_argument("--output_dir", type=str, default="auto",
                        help="Directory to save the converted checkpoint.")

    main(parser.parse_args())
