# Mutational Effect Transfer Learning
[![GitHub Actions](https://github.com/gitter-lab/metl/actions/workflows/test.yaml/badge.svg)](https://github.com/gitter-lab/metl/actions/workflows/test.yaml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10819483.svg)](https://zenodo.org/doi/10.5281/zenodo.10819483)

This repository contains the Mutational Effect Transfer Learning (METL) framework for pretraining and finetuning biophysics-informed protein language models. 
You can use it to train models on your own data or recreate the results from our manuscript.
This framework uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). 

- To run our pretrained METL [models](https://zenodo.org/doi/10.5281/zenodo.11051644) locally in pure PyTorch with minimal software dependencies, see our [metl-pretrained](https://github.com/gitter-lab/metl-pretrained) repository.
- To recreate the results from our preprint, see our [metl-pub](https://github.com/gitter-lab/metl-pub) repository and Zenodo [datasets](https://zenodo.org/doi/10.5281/zenodo.10967412).
- To run your own molecular simulations, see our [metl-sim](https://github.com/gitter-lab/metl-sim) repository.
- To generate molecular simluations in the Open Science Pool, see our [instructions](https://github.com/gitter-lab/metl-sim/tree/master/notebooks/osg) and notebook.
- To finetune or generate predictions with pretrained METL models in Colab, see the [notebooks](notebooks) directory for links to Colab notebooks.
- To generate predictions with pretrained METL models in Hugging Face, see the METL [demo](https://huggingface.co/spaces/gitter-lab/METL_demo) and [model card](https://huggingface.co/gitter-lab/METL).

For more information, please see our manuscript:

[Biophysics-based protein language models for protein engineering](https://doi.org/10.1101/2024.03.15.585128).  
Sam Gelman, Bryce Johnson, Chase Freschlin, Sameer D'Costa, Anthony Gitter<sup>+</sup>, Philip A Romero<sup>+</sup>.  
*bioRxiv*, 2024. doi:10.1101/2024.03.15.585128  
<sup>+</sup> denotes equal contribution.

# Installation

Clone this repository and install the required packages using [conda](https://docs.anaconda.com/free/miniconda/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/index.html):
```bash
conda env create -f environment.yml
conda activate metl
```

Installation typically takes approximately 5 minutes. 

For GPU support, make sure you have the appropriate CUDA version installed.
Add `cudatoolkit` to the `environment.yml` file before creating the conda environment.


# Pretraining on Rosetta data

Rosetta pretraining data is stored in the [rosetta_data](data/rosetta_data) directory.
This repository contains a sample Rosetta dataset for [avGFP](data/rosetta_data/avgfp) with 10,000 variants, which can be used to pretrain a toy avGFP METL-Local model.
For more information on how to acquire or create a Rosetta dataset, see the README in the [rosetta_data](data/rosetta_data) directory.

Once you've downloaded or created a Rosetta pretraining dataset, you can pretrain a METL model using [train_source_model.py](code/train_source_model.py).
The notebook [pretraining.ipynb](notebooks/pretraining.ipynb) shows a complete example of how to pretrain a METL model using the sample avGFP dataset.

You can run the pretraining script on the sample dataset using the following command:

```bash
python code/train_source_model.py @args/pretrain_avgfp_local.txt
```

Note this might take a while to train, so for demonstration purposes, you may want to limit the number of epochs and amount of data using the following:

```bash
python code/train_source_model.py @args/pretrain_avgfp_local.txt --max_epochs 5 --limit_train_batches 5 --limit_val_batches 5 --limit_test_batches 5
```

The test metrics are expected to show poor performance after such a short training run.
For instance, `pearson_total_score` may be around 0.24.

Running the limited pretraining demo takes approximately 5 minutes on CPU.

See the help message for an explanation of all the arguments
```bash
python code/train_source_model.py --help
```

# Finetuning on experimental data

Experimental data is stored in [dms_data](data/dms_data) directory. 
For demonstration purposes, this repository contains the avGFP experimental dataset from [Sarkisyan et al. (2016)](https://doi.org/10.1038/nature17995). 
See the [metl-pub](https://github.com/gitter-lab/metl-pub) repository to access the other experimental datasets we used in our manuscript.
See the README in the [dms_data](data/dms_data) directory for information about how to use your own experimental dataset. 

In addition to experimental data, you will need a pretrained METL model to finetune.
You can pretrain METL models yourself using this repository, or you can use our pretrained METL models from the [metl-pretrained](https://github.com/gitter-lab/metl-pretrained) repository. 

Once you have a pretrained METL model and an experimental dataset, you can finetune the model using [train_target_model.py](code/train_target_model.py).
The notebook [finetuning.ipynb](notebooks/finetuning.ipynb) shows a complete example of how to finetune a METL model using the sample avGFP dataset.
For demonstration purposes, it uses the command:
```bash
python code/train_target_model.py @args/finetune_avgfp_local.txt --enable_progress_bar false --enable_simple_progress_messages --max_epochs 50 --unfreeze_backbone_at_epoch 25
```

Following the short demonstration pretraining and finetuning process is expected to give test set Spearman correlation around 0.6.

Running the finetuning demo takes approximately 7 minutes on CPU.

See the help message for an explanation of all the arguments
```bash
python code/code/train_target_model.py --help
```
