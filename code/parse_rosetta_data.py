""" generate a rosetta source dataset from database by applying a query to database and saving results as csv
    also contains other utility functions for dealing with source datasets, etc """

import os
from os.path import join, basename
import time
import logging
from typing import Optional, Union
import argparse

import connectorx
import numpy as np
import pandas as pd
from pandas.io.sql import SQLiteDatabase, SQLiteTable
import sqlite3
from tqdm import tqdm

try:
    from . import constants
    from . import utils
    from . import rosetta_data_utils as rd
except ImportError:
    import constants
    import utils
    import rosetta_data_utils as rd

logger = logging.getLogger("METL." + __name__)
logger.setLevel(logging.DEBUG)


def gen_dataset_from_query(ds_name: str,
                           pdb_fns: Union[str, list[str], tuple[str]],
                           db_fn: str,
                           keep_num_muts: Optional[list[int]] = None,
                           remove_nan: bool = True,
                           remove_duplicates: bool = True,
                           remove_outliers: bool = True,
                           outlier_energy_term: str = "total_score",
                           outlier_threshold: float = 6.5,
                           replace_pdb_fn: Optional[str] = None,
                           ct_fn: str = "data/rosetta_data/create_tables.sql",
                           save_csv: bool = True,
                           save_sql: bool = True,
                           save_hdf_fixed: bool = True):

    """ generate a rosetta dataset by querying the main variant database from rosettafy """

    # get list of PDB fns to include in this dataset
    if pdb_fns.endswith(".txt"):
        pdb_fns = utils.load_lines(pdb_fns)
    elif not isinstance(pdb_fns, list) and not isinstance(pdb_fns, tuple):
        pdb_fns = [pdb_fns]

    # create output directory
    base_data_dir = join("data/rosetta_data", ds_name)
    data_dir = utils.find_next_sequential_dirname(base_data_dir)

    logger.info("output data directory will be: {}".format(data_dir))
    utils.mkdir(data_dir)

    # database access
    logger.info("connecting to database at: {}".format(db_fn))
    con = sqlite3.connect(db_fn)
    cur = con.cursor()

    # query the database for how many variants are in it (for info.txt)
    count_query = "SELECT COUNT(*) from variant"
    db_count = cur.execute(count_query).fetchall()[0][0]
    print("db_count: {}".format(db_count))

    # query the database for what HTCondor runs are included in it (for info.txt)
    # the "cluster" column in the job table uniquely identifies each condor run
    process_query = "SELECT DISTINCT cluster FROM job"
    process_list = cur.execute(process_query).fetchall()
    process_list = [pl[0] for pl in process_list]
    print("process_list: {}".format(process_list))

    # query the database for what HTCondor runs will be represented in the final dataset
    # note: this should match the query below
    process_query_ds = "SELECT DISTINCT(job.cluster) FROM variant " \
                       "INNER JOIN job ON variant.job_uuid=job.uuid " \
                       "WHERE variant.pdb_fn IN ({})".format(', '.join('?' for _ in pdb_fns))

    process_list_ds = cur.execute(process_query_ds, pdb_fns).fetchall()
    process_list_ds = [pl[0] for pl in process_list_ds]
    print("process_list_ds: {}".format(process_list_ds))

    # sqlite3 close
    cur.close()
    con.close()

    # connectorx will parallelize all the queries and join at the end
    queries = ["SELECT * FROM variant WHERE `pdb_fn` == \"{}\"".format(pdb_fn) for pdb_fn in pdb_fns]
    conn = f"sqlite://{db_fn}"
    db: pd.DataFrame = connectorx.read_sql(conn, queries, return_type="pandas")
    print("initial data loaded into dataframe")

    # number of variants in the pandas dataframe resulting from database query
    df_count = db.shape[0]

    # filter by number of mutations
    num_variants_after_num_muts_filter = db.shape[0]
    if keep_num_muts is not None:
        print("Filtering variants by number of mutations")
        start_len = len(db)
        # vectorize it
        keep_num_muts = np.array(keep_num_muts)
        num_muts = db["mutations"].apply(lambda x: len(x.split(",")))
        db = db[num_muts.isin(keep_num_muts)].reset_index(drop=True)
        print("Dropped {} variants by number of mutations".format(start_len - len(db)))
        num_variants_after_num_muts_filter = db.shape[0]
        print("Num variants after num muts filter: {}".format(num_variants_after_num_muts_filter))

    # filter any variants with NAN values
    num_variants_after_nan_filter = db.shape[0]
    if remove_nan:
        print("Filtering variants with NaN values")
        start_len = len(db)
        db = db.dropna(axis=0, how="any").reset_index(drop=True)
        print("Dropped {} variants with nan values".format(start_len - len(db)))
        num_variants_after_nan_filter = db.shape[0]
        print("Num variants after NaN filter: {}".format(num_variants_after_nan_filter))

    # remove duplicates by choosing a random one to keep
    num_variants_after_duplicate_filter = db.shape[0]
    if remove_duplicates:
        # shuffle the dataframe, choose first to keep, then re-sort by index
        db = db.sample(frac=1, random_state=72, replace=False)
        db = db.drop_duplicates(subset=["pdb_fn", "mutations"], keep="first")
        db = db.sort_index(axis=0, ascending=True)
        num_variants_after_duplicate_filter = db.shape[0]
        print("Removed {} duplicates".format(num_variants_after_nan_filter - num_variants_after_duplicate_filter))
        print("Num variants after duplicate filter: {}".format(num_variants_after_duplicate_filter))
        # variants should be back in sorted order with the correct index after the duplicates are removed
        # so no need to reset the index again

    # remove outliers...
    num_variants_after_outlier_removal = db.shape[0]
    if remove_outliers:
        def is_outlier(data, m=6.5):
            # https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            return s > m

        db = db[~db.groupby("pdb_fn", group_keys=False)[outlier_energy_term].apply(
            lambda x: is_outlier(x, m=outlier_threshold))]
        print("Removed {} outliers".format(num_variants_after_duplicate_filter - db.shape[0]))
        num_variants_after_outlier_removal = db.shape[0]
        print("Num variants after outlier removal: {}".format(num_variants_after_outlier_removal))

    # convert int columns to regular non-nullable int
    # not sure if this problem is coming from connectorx or new version of pandas
    # either way, Int64 (nullable) causes problems when saving to hdf5, whereas int64(non-nullable) doesnt
    int_cols = ["run_time",
                "mutate_run_time",
                "relax_run_time",
                "filter_run_time",
                "centroid_run_time",
                "dock_run_time"]

    for int_col in int_cols:
        if int_col in db.columns:
            db[int_col] = db[int_col].astype(int)

    # DLG4 requires some special consideration because the indexing in the PDB file used for Rosetta is different
    # from the DMS dataset / truncated PDB. the indexing offset is in constants.py as rosettafy_pdb_offset.
    # this only need to be checked for local datasets that are defined in constants... not the global DS
    datasets = utils.load_dataset_metadata()
    if ds_name in datasets and len(pdb_fns) == 1:
        if "rosettafy_pdb_offset" in datasets[ds_name]:
            print("Offsetting variant indexing and PDB fn for {}".format(ds_name))
            # note that the indexing offset "rosettafy_pdb_offset" is the exact amount needed to go from
            # the dms 0-based indexing to the rosettafy 1-based indexing. we multiply by -1 to go from
            # the rosettafy 1-based indexing back to the dms 0-based indexing
            indexing_offset = -1 * datasets[ds_name]["rosettafy_pdb_offset"]
            # we also need to add 1 to the indexing_offset because we are creating indexing for
            # the truncated DMS pdb file, which still needs to be 1-indexed
            indexing_offset += 1
            db["mutations"] = utils.convert_indexing(db["mutations"], indexing_offset)

        # change the PDB file to the DMS one (which should have matching indexing)
        # if rosettafy_pdb_offset is set, then this should also be the case...
        if datasets[ds_name]["pdb_fn"] != datasets[ds_name]["rosettafy_pdb_fn"]:
            print("Changing PDB fn for {} from {} to {}".format(ds_name,
                                                                datasets[ds_name]["pdb_fn"],
                                                                datasets[ds_name]["rosettafy_pdb_fn"]))
            db["pdb_fn"] = basename(datasets[ds_name]["pdb_fn"])

    # for DLG4-2022, the Rosetta dataset name doesn't match the DMS dataset names (-Binding and -Abundance)
    # so for replacing the PDB file, we have a separate argument for replace_pdb_fn
    # if we have more datasets requiring special treatment in the future, may need to re-think this whole approach
    # because it's getting a bit messy
    # one option might be to specify everything as arguments to this function rather than relying on
    # looking up the ds_name in constants
    if replace_pdb_fn is not None:
        print("Replacing PDB fn for {} with {}".format(ds_name, replace_pdb_fn))
        db["pdb_fn"] = basename(replace_pdb_fn)

    # create an info.txt for the dataset
    with open(join(data_dir, "info.txt"), "w") as f:
        f.write("dataset created by querying database\n")
        f.write("time,{}\n".format(time.strftime("%Y-%m-%d_%H-%M-%S")))
        f.write("db_fn,{}\n".format(db_fn))
        f.write("condor_runs_in_db,{}\n".format(process_list))
        f.write("num_variants_in_db,{}\n".format(db_count))
        f.write("query,\"{}\"\n".format("\n".join(queries)))
        f.write("pdb_fns,{}\n".format(",".join(pdb_fns)))
        f.write("condor_runs_in_query_results,{}\n".format(process_list_ds))
        f.write("num_variants_from_query,{}\n".format(df_count))
        if keep_num_muts is not None:
            f.write("filtered,num_muts\n")
            f.write("num_variants_after_num_muts_filter,{}\n".format(num_variants_after_num_muts_filter))
        if remove_nan:
            f.write("filtered,nans\n")
            f.write("num_variants_after_nan_filter,{}\n".format(num_variants_after_nan_filter))
        if remove_duplicates:
            f.write("filtered,duplicates (choose random duplicate to keep)\n")
            f.write("num_variants_after_duplicate_filter,{}\n".format(num_variants_after_duplicate_filter))
        if remove_outliers:
            f.write("filtered,outliers ({} > {})\n".format(outlier_energy_term, outlier_threshold))
            f.write("num_variants_after_outlier_removal,{}\n".format(num_variants_after_outlier_removal))

    save_ds_to_files(db,
                     data_dir,
                     ds_name,
                     ct_fn=ct_fn,
                     save_csv=save_csv,
                     save_sql=save_sql,
                     save_hdf_fixed=save_hdf_fixed)

    # print(db.head())


def gen_dataset_dms_cov(ds_name, db_fn=None):
    # note: this does NOT filter out variants with NaN because even if some of the DMS variants
    # have NaNs for Rosetta terms, we still want to include them in this full coverage dataset

    # create output directory
    ds_cov_name = "{}_dms_cov".format(ds_name)
    data_dir = utils.find_next_sequential_dirname(join("data/rosetta_data/dms_coverage", ds_cov_name))

    print("output data directory will be: {}".format(data_dir))
    utils.mkdir(data_dir)

    rosetta_ds = rd.get_rosetta_ds(ds_name, db_fn=db_fn, assert_coverage=True)

    # save dataset to files
    save_ds_to_files(rosetta_ds, data_dir, ds_cov_name)

    # write info file
    with open(join(data_dir, "info.txt"), "w") as f:
        f.write("dataset created by querying database to get full coverage of {} DMS dataset\n".format(ds_name))
        f.write("some variants might have NaN values (see comments in code)\n")
        f.write("time,{}\n".format(time.strftime("%Y-%m-%d_%H-%M-%S")))


def create_blank_db(db_fn: str,
                    ct_fn: str = "data/rosetta_data/create_tables.sql"):

    # retrieve table creation commands from file
    with open(ct_fn, "r") as f:
        sql_commands_str = f.read()
    sql_commands = sql_commands_str.split(';')

    # create cursor to interact with database connection
    con = sqlite3.connect(db_fn)
    cur = con.cursor()
    # run the table creation commands
    for command in sql_commands:
        cur.execute(command)
    con.commit()
    con.close()


def df_to_sqlite(df: pd.DataFrame, db_file_name: str, table_name: str, chunk_size: int = 1000000):
    # https://stackoverflow.com/a/70488765/227755
    # https://stackoverflow.com/questions/56369565/large-6-million-rows-pandas-df-causes-memory-error-with-to-sql-when-chunksi
    con = sqlite3.connect(db_file_name)
    db = SQLiteDatabase(con=con)
    table = SQLiteTable(table_name, db, df, index=False, if_exists="append", dtype=None)
    table.create()
    insert = table.insert_statement(num_rows=1)  # single insert statement
    it = df.itertuples(index=False, name=None)  # just regular tuples
    pb = tqdm(it, total=len(df))  # not needed but nice to have
    with con:
        while True:
            con.execute("begin")
            try:
                for c in range(0, chunk_size):
                    row = next(it, None)
                    if row is None:
                        pb.update(c)
                        return
                    con.execute(insert, row)
                pb.update(chunk_size)
            finally:
                con.execute("commit")


def save_ds_to_files(df: pd.DataFrame,
                     save_dir: str,
                     save_fn_base: str,
                     ct_fn: str = "data/rosetta_data/create_tables.sql",
                     save_csv: bool = True,
                     save_sql: bool = True,
                     save_hdf_fixed: bool = True):

    """ saves datasets to files... note, the save_dir must already exist """

    # save to csv
    if save_csv:
        print("Saving dataset to CSV")
        csv_fn = join(save_dir, save_fn_base + ".tsv")
        df.to_csv(csv_fn, sep="\t", index=False)

    # save to hdf - PANDAS FIXED FORMAT
    if save_hdf_fixed:
        print("Saving dataset to HDF, pandas fixed format")
        hdf_fn = join(save_dir, save_fn_base + ".h5")
        df.to_hdf(hdf_fn, key="variant", format="fixed")

    # save to sql
    if save_sql:
        print("Saving dataset to SQL")
        db_fn = join(save_dir, save_fn_base + ".db")
        create_blank_db(db_fn, ct_fn)
        df_to_sqlite(df, db_fn, "variant")

    # save pdb fn list
    pdb_fn_list_fn = join(save_dir, "pdb_fns.txt")
    utils.save_lines(pdb_fn_list_fn, df["pdb_fn"].to_list())


def gen_dataset_from_dataset():
    """ generate a dataset from a different dataset rather than the database
        mostly for speeding up subsampled dataset creation for testing purposes """

    # base dataset
    ds_fn = "data/rosetta_data/gb1/gb1.h5"
    # number of variants to sample
    num_to_sample = 200000

    # random seed
    random_seed = 7
    # new dataset name
    ds_name = "gb1_sample"
    # data output directory
    data_dir = join("data/rosetta_data", ds_name)

    # create output directory
    os.makedirs(data_dir)

    # load the dataset and sample
    ds = pd.read_hdf(ds_fn, key="variant")
    sampled_ds = ds.sample(n=num_to_sample, replace=False, random_state=random_seed, axis=0)

    save_ds_to_files(sampled_ds, data_dir, ds_name)

    # write info file
    with open(join(data_dir, "info.txt"), "w") as f:
        f.write("dataset created by sampling a different dataset\n")
        f.write("time,{}\n".format(time.strftime("%Y-%m-%d_%H-%M-%S")))
        f.write("parent_dataset,{}\n".format(ds_fn))
        f.write("num_to_sample,{}\n".format(num_to_sample))
        f.write("random_seed,{}\n".format(random_seed))


def combine_datasets(standard_fn, docking_fn, ds_name="gb1-de-standard-docking"):
    """ combining datasets to create GB1-IgG dataset with energy terms from both runs """

    # for the combined standard and docking dataset
    base_data_dir = join("data/rosetta_data", ds_name)
    data_dir = utils.find_next_sequential_dirname(base_data_dir)
    logger.info("output data directory will be: {}".format(data_dir))
    utils.mkdir(data_dir)

    # load datasets
    if standard_fn.endswith(".h5"):
        standard_df: pd.DataFrame = pd.read_hdf(standard_fn, key="variant")
    elif standard_fn.endswith(".tsv"):
        standard_df = pd.read_csv(standard_fn, sep="\t")
    else:
        raise ValueError("Unexpected standard dataset file format: {}".format(standard_fn))

    if docking_fn.endswith(".h5"):
        docking_df: pd.DataFrame = pd.read_hdf(docking_fn, key="variant")
    elif docking_fn.endswith(".tsv"):
        docking_df = pd.read_csv(docking_fn, sep="\t")
    else:
        raise ValueError("Unexpected docking dataset file format: {}".format(docking_fn))

    # for purposes of combined dataset, only include columns which will be used in training
    # this is partly to avoid having to deal with duplicate columns (e.g. total_score)
    standard_target_cols = constants.ROSETTA_ATTRIBUTES_TRAINING
    docking_target_cols = constants.ROSETTA_ATTRIBUTES_DOCKING_TRAINING

    # raise an error if there are duplicate columns
    if len(set(standard_target_cols).intersection(set(docking_target_cols))) > 0:
        # if we want to support this in the future, will need to rename the columns
        # for now it's just acting as a sanity check as we shouldn't have this happen
        raise ValueError("Duplicate energies found in standard and docking target energies")

    # drop all columns that aren't in the target columns
    # this will also drop all the run_time columns which is okay for this dataset
    keep_cols = ["pdb_fn", "mutations", "job_uuid"]
    standard_df_tc = standard_df.drop(
        columns=[col for col in standard_df.columns if col not in list(standard_target_cols) + keep_cols])
    docking_df_tc = docking_df.drop(
        columns=[col for col in docking_df.columns if col not in list(docking_target_cols) + keep_cols])

    # the PDBs in the two datasets do not match because the standard dataset uses only the GB1 structure
    # whereas the docking dataset uses the GB1-IgG complex structure... for this dataset, all should be the same
    # we need just the GB1 structure alone, so just copy over the structure from ds_1
    # note... we may have handled this upstream so in that case don't do anything
    standard_pdb = standard_df_tc["pdb_fn"].iloc[0]
    docking_pdb = docking_df_tc["pdb_fn"].iloc[0]
    if docking_pdb != standard_pdb:
        docking_df_tc["pdb_fn"] = standard_pdb

    # the job UUID column name needs to be unique because we want to keep track of which jobs both the
    # standard and docking energies come from
    standard_df_tc = standard_df_tc.rename(columns={"job_uuid": "job_uuid_standard"})
    docking_df_tc = docking_df_tc.rename(columns={"job_uuid": "job_uuid_docking"})

    # merge the two datasets
    combined_ds = standard_df_tc.merge(docking_df_tc, how="inner", on=["pdb_fn", "mutations"]).reset_index(drop=True)

    # move the job_uuid_docking column to be right after the job_uuid_standard column
    # this is just for convenience so that the job_uuids are next to each other
    cols = list(combined_ds.columns)
    cols.insert(cols.index("job_uuid_standard") + 1, cols.pop(cols.index("job_uuid_docking")))
    combined_ds = combined_ds[cols]

    # print out how many variants in each dataset
    print("standard_df_tc: {}".format(standard_df_tc.shape[0]))
    print("docking_df_tc: {}".format(docking_df_tc.shape[0]))
    print("combined_ds: {}".format(combined_ds.shape[0]))

    # save the combined dataset
    # make sure to specify the ct_fn (create tables) so that the database is set up for the
    # correct energy terms (i.e. the ones in the combined dataset)
    ct_fn = "data/rosetta_data/create_tables_standard_docking.sql"
    save_ds_to_files(combined_ds, data_dir, ds_name, ct_fn=ct_fn)


def main(args):
    # need this to output logging calls
    logging.basicConfig()

    if args.mode == "generate_dataset":
        gen_dataset_from_query(ds_name=args.ds_name,
                               pdb_fns=args.pdb_fns,
                               db_fn=args.db_fn,
                               keep_num_muts=args.keep_num_muts,
                               remove_nan=args.remove_nan,
                               remove_duplicates=args.remove_duplicates,
                               remove_outliers=args.remove_outliers,
                               outlier_energy_term=args.outlier_energy_term,
                               outlier_threshold=args.outlier_threshold,
                               replace_pdb_fn=args.replace_pdb_fn,
                               ct_fn=args.ct_fn)

    elif args.mode == "generate_dms_coverage_dataset":
        gen_dataset_dms_cov(ds_name=args.ds_name, db_fn=args.db_fn)
        # ds_name = "gb1"
        # db_fn = "/Users/sg/PycharmProjects/rosettafy/variant_database/gb1-docking.db"
        # gen_dataset_dms_cov(ds_name, db_fn=db_fn)

    elif args.mode == "combine_datasets":
        # this was used to create the GB1-IgG dataset with energies from both the standard and binding runs
        # standard_fn = "data/rosetta_data/gb1_2/gb1.h5"
        # docking_fn = "data/rosetta_data/gb1-docking-all/gb1-docking-all.h5"
        # combine_datasets(standard_fn, docking_fn)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument("mode",
                        help="mode to run",
                        type=str,
                        choices=["generate_dataset", "generate_dms_coverage_dataset"],
                        default="generate_dataset")

    parser.add_argument("--ds_name",
                        help="name of the dataset to generate",
                        type=str)

    parser.add_argument("--pdb_fns",
                        help="either the path to a single pdb file or path to a file containing a list of "
                             "pdb filenames to include in the dataset",
                        type=str)

    parser.add_argument("--db_fn",
                        help="path to the variant database created in metl-sim",
                        type=str)

    parser.add_argument("--keep_num_muts",
                        help="list of number of mutations to keep (for example, to create a singles-only dataset)",
                        default=None,
                        nargs='+')

    parser.add_argument('--remove_nan', action='store_true',
                        help="remove any variants with NaN values (default)")
    parser.add_argument('--no_remove_nan', dest='remove_nan', action='store_false')
    parser.set_defaults(remove_nan=True)

    parser.add_argument('--remove_duplicates', action='store_true',
                        help="remove duplicate variants (default)")
    parser.add_argument('--no_remove_duplicates', dest='remove_duplicates', action='store_false')
    parser.set_defaults(remove_duplicates=True)

    parser.add_argument('--remove_outliers', action='store_true',
                        help="remove outliers using median absolute deviation method (default)")
    parser.add_argument('--no_remove_outliers', dest='remove_outliers', action='store_false')
    parser.set_defaults(remove_outliers=True)

    parser.add_argument("--outlier_energy_term",
                        help="the energy term to use for outlier removal",
                        type=str,
                        default="total_score")

    parser.add_argument("--outlier_threshold",
                        help="the threshold for outlier removal",
                        type=float,
                        default=6.5)

    parser.add_argument("--replace_pdb_fn",
                        help="replace the PDB filename in the dataset with this filename",
                        type=str,
                        default=None)

    parser.add_argument("--ct_fn",
                        help="path to the SQL create tables file",
                        type=str,
                        default="data/rosetta_data/create_tables.sql")

    main(parser.parse_args())
