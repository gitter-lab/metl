""" parse source data into simple tsv datasets. """
import os
from os.path import isfile, join, dirname, isdir
import argparse

import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils


def sort_and_save_to_csv(df, out_fn, precision=7, sort_muts=True, sort_variants=True, na_rep=""):
    # make sure variants have their list of mutations in sorted order
    if sort_muts:
        df["variant"] = utils.sort_variant_mutations(df["variant"])

    # put variants in sorted order by number of mutations, location of mutations, amino acids
    if sort_variants:
        sorted_variants = utils.sort_variants(df["variant"])
        df = df.sort_values(by="variant", key=lambda vs: vs.map({v: k for k, v in enumerate(sorted_variants)}))

    float_format = "{:." + str(precision) + "f}"

    if not isdir(dirname(out_fn)):
        os.makedirs(dirname(out_fn))

    df.to_csv(out_fn, sep="\t", float_format=float_format.format, index=False, na_rep=na_rep)


def parse_avgfp(score_precision=7):
    """ create the gfp dataset from raw source data """
    source_fn = "data/raw_dms_data/avgfp/amino_acid_genotypes_to_brightness.tsv"
    out_fn = "data/dms_data/avgfp/avgfp.tsv"
    if isfile(out_fn):
        print("err: parsed avgfp dataset already exists: {}".format(out_fn))
        return

    # load the source data
    data = pd.read_csv(source_fn, sep="\t")

    # remove the wild-type entry
    data = data.loc[1:]

    # create columns for variants, number of mutations, and score
    variants = data["aaMutations"].apply(lambda x: ",".join([x[1:] for x in x.split(":")]))
    num_mutations = variants.apply(lambda x: len(x.split(",")))
    score = data["medianBrightness"]

    # create the dataframe
    cols = ["variant", "num_mutations", "score"]
    data_dict = {"variant": variants.values, "num_mutations": num_mutations.values, "score": score.values}
    df = pd.DataFrame(data_dict, columns=cols)

    # normalize the score so WT=0 by subtracting the WT score
    df["score"] = df["score"].apply(lambda x: np.round(x - 3.7192121319, score_precision))

    # drop variants with stop codon mutations
    df = df[~df["variant"].str.contains(r"\*")].reset_index(drop=True)

    sort_and_save_to_csv(df, out_fn, precision=score_precision)


def main(args):
    parse_funcs = {
        # dictionary of datasets and the corresponding parsing functions
        "avgfp": parse_avgfp
    }

    if args.ds_name not in parse_funcs:
        print("err: dataset name not recognized: {}".format(args.ds_name))
        return

    parse_funcs[args.ds_name]()


if __name__ == "__main__":
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name",
                        help="name of the dataset to parse",
                        type=str)
    main(parser.parse_args())
