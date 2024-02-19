# Experimental data

Each dataset has its own directory containing the following:
- The dataset .tsv file, a tab-separated dataframe with columns for the variant, number of mutations, and functional score. 
- A splits directory containing train, validation, and test splits that have been saved as files for reuse and reproducibility. See the notebook [train_test_split.ipynb](../notebooks/train_test_split.ipynb) for an example of how to create these splits.

Additionally, there is a dataset index file, [datasets.yml](datasets.yml), with additional information about each dataset, such as the name, directory, and wild-type sequence.

## Using your own dataset

To use your own dataset, you will need to format it using our standard format and add an entry for it to [datasets.yml](datasets.yml).
I recommend placing raw data in the [data/raw_dms_data](../raw_dms_data) directory and adding a function to the [parse_raw_dms_data.py](../code/parse_raw_dms_data.py) script to process the dataset into the required format, but this is not strictly necessary (just good practice).

