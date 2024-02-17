""" general utility functions used throughput codebase """
import os
from os.path import isfile, isdir
from typing import Optional
from io import StringIO

import pandas as pd
import yaml
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser

import constants


def mkdir(d):
    """ creates given dir if it does not already exist """
    if not isdir(d):
        os.makedirs(d)


def load_lines(fn: str):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines


def save_lines(out_fn: str, lines: list):
    """ saves each line in data to given file """
    with open(out_fn, "w") as f_handle:
        for line in lines:
            f_handle.write("{}\n".format(line))


def sort_variant_mutations(variants):
    """ put variant mutations in sorted order by position """
    sorted_variants = []
    for variant in variants:
        muts = variant.split(",")
        positions = [int(mut[1:-1]) for mut in muts]
        # now sort muts by positions index, then join on "," to recreate variant
        sorted_muts = [x for x, _ in sorted(zip(muts, positions), key=lambda pair: pair[1])]
        sorted_variants.append(",".join(sorted_muts))
    return sorted_variants


def sort_variants(variants):
    """ sort variants by number of mutations and mutation positions
        used to get a master order for sorting the dataframe """

    # first sort by number of mutations
    def key_func_1(v):
        return len(v.split(","))

    # next sort by positions of the mutations
    def key_func_2(v):
        return [int(x[1:-1]) for x in v.split(",")]

    # finally sort by the amino acid order defined in constants.CHARS
    def key_func_3(v):
        return [constants.CHARS.index(x[-1]) for x in v.split(",")]

    sv = sorted(variants, key=lambda v: [key_func_1(v), key_func_2(v), key_func_3(v)])

    return sv


def load_dataset_metadata(metadata_fn: str = "data/dms_data/datasets.yml"):
    with open(metadata_fn, "r") as stream:
        try:
            datasets = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return datasets


def load_dataset(ds_name: Optional[str] = None,
                 ds_fn: Optional[str] = None,
                 sort_mutations: bool = False,
                 load_epistasis: bool = False):
    """ load a dataset as pandas dataframe """
    if ds_name is None and ds_fn is None:
        raise ValueError("must provide either ds_name or ds_fn to load a dataset")

    if ds_fn is None:
        datasets = load_dataset_metadata()
        ds_fn = datasets[ds_name]["ds_fn"]

    if not isfile(ds_fn):
        raise FileNotFoundError("can't load dataset, file doesn't exist: {}".format(ds_fn))

    ds = pd.read_csv(ds_fn, sep="\t")

    # ensure variants are in sorted order
    if sort_mutations and "variant" in ds:
        ds["variant"] = sort_variant_mutations(ds["variant"])

    # load epistasis data if it exists
    epistasis_fn = ds_fn.replace(".tsv", "_epistasis.tsv")
    if load_epistasis and isfile(epistasis_fn):
        epistasis_df = pd.read_csv(epistasis_fn, sep="\t")
        # merge epistasis data with dataset
        ds = pd.merge(ds, epistasis_df, on="variant", how="left")

    return ds


def convert_indexing(variants, offset):
    """ convert between 0-indexed and 1-indexed """
    converted_to_list = False
    if not isinstance(variants, list) and not isinstance(variants, tuple) and not isinstance(variants, pd.Series):
        converted_to_list = True
        variants = [variants]

    converted = [",".join(["{}{}{}".format(mut[0], int(mut[1:-1]) + offset, mut[-1])
                           for mut in v.split(",")])
                 for v in variants]

    if converted_to_list:
        converted = converted[0]

    return converted


def find_next_sequential_dirname(base_dir):
    """ finds the next sequential dirname by appending _i, where i is the dir number """
    next_dir = base_dir
    if isdir(base_dir):
        i = 2
        next_dir = base_dir + "_{}".format(i)
        while isdir(next_dir):
            i += 1
            next_dir = base_dir + "_{}".format(i)
    return next_dir


def clean_pdb_data(pdb_fn):
    """ only keep ATOM and HETATM lines """
    with open(pdb_fn, 'r') as fin:
        lines = [line for line in fin if line.startswith(('ATOM', 'HETATM'))]
    return ''.join(lines)


def extract_seq_from_pdb(pdb_fn: str,
                         chain_id: Optional[str] = None,
                         error_on_missing_residue: bool = True,
                         error_on_multiple_chains: bool = True):

    """
    Extract the sequence from a PDB file
    :param pdb_fn: path to PDB file
    :param chain_id: If None, return sequences for every chain in PDB. Otherwise, extract sequence only for given chain
    :param error_on_missing_residue: if True, raise an error if a residue is missing, based on sequential numbering
    :param error_on_multiple_chains: if True and chain_id is None, raise error if the PDB contains more than one chain
    :return: the seq as a str (if returning 1 chain) or a dict mapping chain id to seq (if returning more than 1 chain)
    """

    # only keep ATOM and HETATM lines
    pdb_data = clean_pdb_data(pdb_fn)
    pdb_handle = StringIO(pdb_data)

    # create a structure object for this PDB
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("structure", pdb_handle)

    # this function only handles PDBs with 1 model
    if len(list(structure.get_models())) > 1:
        raise ValueError("PDB contains more than one model")

    # retrieve the chains in this PDB
    chains = list(structure.get_chains())
    valid_chain_ids = [chain.id for chain in chains]
    if chain_id is not None:
        if chain_id not in valid_chain_ids:
            raise ValueError(
                "Invalid chain_id '{}' for PDB file which contains chains {}".format(chain_id, valid_chain_ids))
        else:
            chains = [chain for chain in chains if chain.id == chain_id]

    if len(chains) > 1 and error_on_multiple_chains:
        raise ValueError("PDB contains more than one chain: {}".format(pdb_fn))

    sequences = []
    for chain in chains:
        # find missing residues (looking for non-sequential residue numberings)
        residue_numbers = [res.id[1] for res in chain if PDB.is_aa(res)]
        missing_residues = []
        if len(residue_numbers) > 1:
            for i in range(1, len(residue_numbers)):
                if residue_numbers[i] - residue_numbers[i - 1] != 1:
                    missing_residues.extend(range(residue_numbers[i - 1] + 1, residue_numbers[i]))

        # error out on a missing residue if requested
        if len(missing_residues) > 0 and error_on_missing_residue:
            raise ValueError("Missing residues {} in chain {}".format(missing_residues, chain.id))

        # build up the sequence
        seq = []
        last_residue_number = None
        for res in [r for r in chain if PDB.is_aa(r)]:
            # fill in missing residues with 'X'
            if last_residue_number is not None:
                while res.id[1] > last_residue_number + 1:
                    seq.append("X")
                    last_residue_number += 1

            seq.append(PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(res.get_resname())))
            last_residue_number = res.id[1]

        sequences.append("".join(seq))

    if len(chains) > 1:
        # for multiple chains, return a dictionary mapping chain id to the sequence
        return {ch.id: seq for ch, seq in zip(chains, sequences)}
    else:
        return sequences[0]