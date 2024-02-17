# Raw experimental data

This directory is for storing raw experimental data downloaded from databases, paper supplements, etc.
This data is processed into a uniform format that can be used to train models with our codebase.
The processed data is contained in the [data/dms_data](../dms_data) directory. 
The [script we used to process the data](../../code/parse_raw_dms_data.py) is provided for reference, but you will need to modify it to work with your own data.

For purposes of this repository, we are only including the `avgfp` dataset as an example.

| Dataset                         | Reference                                                                                                                                      | First Author | Year | Acquired From                                                       | Link                                                                                                                                                                                                                                                   |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------|------|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| avgfp                           | [Local fitness landscape of the green fluorescent protein](https://doi.org/10.1038/nature17995)                                                | Sarkisyan    | 2016 | Associated data on figshare, amino_acid_genotypes_to_brightness.tsv | [Figshare](http://dx.doi.org/10.6084/m9.figshare.3102154), [Direct download](https://figshare.com/ndownloader/files/4820647)                                                                                                                           |

You can process this raw dataset by running the following command from the root of the repository.
Note you may have to delete the `avgfp.tsv` in `data/dms_data/avgfp` if it already exists, in order to get this script to run.

```bash
python code/parse_raw_dms_data.py avgfp
```