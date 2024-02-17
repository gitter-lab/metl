# Mutational Effect Transfer Learning
This repository contains the Mutational Effect Transfer Learning (METL) framework for pretraining and finetuning biophysics-informed protein language models. 
You can use it to train models on your own data or recreate the results from our manuscript.
This framework uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). 

- To access pretrained METL models in pure PyTorch with minimal software dependencies, see our [metl-pretrained](url) repository.
- To recreate the results from our preprint, see our [metl-pub](url) repository.
- To run your own molecular simulations, see our [rosettafy](url) repository.

For more information, please see our [preprint](url).

# Installation

Clone this repository and install the required packages using [conda](https://docs.anaconda.com/free/miniconda/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/index.html):
```bash
conda env create -f environment.yml
conda activate metl
```

For GPU support, make sure you have the appropriate CUDA version installed.


# Pretraining on Rosetta data

Rosetta pretraining data is stored in the [rosetta_data](data/rosetta_data) directory.
This repository contains a sample Rosetta dataset for [avGFP](data/rosetta_data/avgfp), which can be used to pretrain a toy avGFP METL-Local model.
For more information on how to acquire or create a Rosetta dataset, see the README in the [rosetta_data](data/rosetta_data) directory.

Once you've downloaded or created a Rosetta pretraining dataset, you can pretrain a METL model using...

# Finetuning on experimental data

Experimental data is stored in [dms_data](data/dms_data) directory. 
For demonstration purposes, this repository contains the avGFP experimental dataset from [Sarkisyan et al. (2016)](https://doi.org/10.1038/nature17995). 
See the [metl-pub](url) repository to access the other experimental datasets we used in our preprint.
See the README in the [dms_data](data/dms_data) directory for information about how to use your own experimental dataset. 

In addition to experimental data, you will need a pretrained METL model to finetune.
You can pretrain METL models yourself using this repository, or you can use our pretrained METL models from the [metl-pretrained](url) repository. 

Once you have a pretrained METL model and an experimental dataset, you can finetune the model using...

# Additional information

Training baseline models, etc.