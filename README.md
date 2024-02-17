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

# Rosetta data



# Experimental data



# Pretraining on Rosetta data


# Finetuning on experimental data