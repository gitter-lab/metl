""" a few miscellaneous functions to help with rosetta datasets """

import pandas as pd
import sqlalchemy as sqla

import utils


def convert_dms_to_rosettafy_indexing(ds_name, variants, reverse=False):
    """ convert from DMS indexing (0-based indexing into the wt_aa defined in constants)
        to rosettafy PDB indexing (1-based indexing into the PDB file used by rosettafy)
        if reverse==True, will instead convert from rosettafy indexing to DMS indexing """

    # default is to just convert from 0-based indexing to 1-based indexing
    indexing_offset = 1

    # DLG4 is a special case where the wt_aa is offset from the sequence in the PDB file used by rosettafy
    # therefore, simply converting 0-based indexing to 1-based indexing won't give correct index to rosettafy PDB file
    # the correct offset in this case is stored in "rosettafy_pdb_offset" in constants.py
    datasets = utils.load_dataset_metadata()
    if "rosettafy_pdb_offset" in datasets[ds_name]:
        indexing_offset = datasets[ds_name]["rosettafy_pdb_offset"]

    # reverse to go from rosettafy indexing to DMS indexing
    if reverse:
        indexing_offset *= -1

    variants = utils.convert_indexing(variants, indexing_offset)

    return variants


def query_database(db_fn, pdb_fn, variants, chunk_size=150000):
    # access the full database from rosettafy

    engine = sqla.create_engine('sqlite:///{}'.format(db_fn))
    conn = engine.connect().execution_options(stream_results=True)

    def chunker(seq, size):
        # https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    chunk_dfs = []
    for variants_chunk in chunker(variants, chunk_size):
        subquery = ",".join("?" * len(variants_chunk))
        query = "SELECT * FROM variant WHERE pdb_fn == ? AND mutations IN ({})".format(subquery)

        chunk_df = pd.read_sql_query(query, conn, params=[pdb_fn] + variants_chunk, coerce_float=False)
        chunk_dfs.append(chunk_df)

    # sqlalchemy close
    conn.close()
    engine.dispose()

    rosetta_ds = pd.concat(chunk_dfs, axis=0).reset_index(drop=True)

    return rosetta_ds


def get_rosetta_ds(ds_name: str,
                   db_fn: str,
                   assert_coverage: bool = True):
    """ load rosetta dataset from the variant database that corresponds to the given dms dataset name
        meant more to be used to save the rosetta dataset to disk because this could return different
        outputs depending on what variants are in the database (for example if duplicates are added)"""

    dms_ds = utils.load_dataset(ds_name)
    variants_rosetta_compatible = convert_dms_to_rosettafy_indexing(ds_name, dms_ds["variant"])

    datasets = utils.load_dataset_metadata()
    pdb_fn = datasets[ds_name]["rosettafy_pdb_fn"]
    rosetta_ds = query_database(db_fn, pdb_fn, variants_rosetta_compatible)

    # drop duplicates
    rosetta_ds = rosetta_ds.sample(frac=1, random_state=72, replace=False)
    rosetta_ds = rosetta_ds.drop_duplicates(subset="mutations", keep="first")
    rosetta_ds = rosetta_ds.sort_index(axis=0, ascending=True)

    # now sort the rosetta_ds with dms coverage to be in the exact same order as the DMS dataset
    # create dictionary for more efficient generation of index for sorting
    sorting_index = {variant: index for index, variant in enumerate(variants_rosetta_compatible)}
    rosetta_ds = rosetta_ds.sort_values(by="mutations", key=lambda vs: [sorting_index[v] for v in vs])

    rosetta_ds = rosetta_ds.reset_index(drop=True)

    if assert_coverage:
        # just a final sanity check to make sure we have full coverage of the DMS dataset
        assert(len(set(variants_rosetta_compatible) - set(rosetta_ds["mutations"])) == 0)

    return rosetta_ds


def main():
    pass


if __name__ == "__main__":
    main()
