# R252 Project (sgm48)

## Summary
The projects core is to reimplement 'Learning Python Code Suggestion with a Sparse Pointer Network' by Bhoopchand et al. (https://arxiv.org/pdf/1611.08307.pdf)
## Overview of the Repository
The main code is in `src`, which is based on the language model provided in class.
It incorporates the model described by Bhoopchand et al. in `src/language_model`.
`src/container` is a directory to build the container that I use with the script `coda.py` to run the training on codalabs GPUs.
All code is compatible for Python version >=3.5.

## Dataset Preparation
As base dataset we use the r252-corpus-features directory provided in class. This can then be split up into subsets however you want and it can be normalized with `src/dataset_normalization.py`. The needed dependencies can be installed with `pip install -r src/normalization_requirements.txt`. Run `python3 src/dataset_normalizer.py --help` to see how to use the normalization script. This script writes to a file called `minimumleft.json` which can then be fed to it again for another run, this counts how many tokens are unused and thus can be used in the next run to reduce the number of tokens to a minimum.

## Running the Language Model
You can run the `src/language_mode` code by using the container provided in `src/container`, that is available on Docker Hub as `samgmuller/tf1.12base:0.4`. If you do not want to use the container you can also install the `src/container/requirements.txt` as well as TensorFlow 1.12. I recommend using a virtual environment to install TensorFlow.
### Training
You can run the training as described by `python3 src/language_model/train.py --help`. The easiest usage would be something like: `python3 src/language_model_train.py SAVE_MODEL_DIR DIR_WITH_TRAIN_DATA DIR_WITH_VALID_DATA`. The training and validation data is assumed to be in the proto format defined by you. 
### Evaluation
The evaluation can be run with `python3 src/language_model/evaluate.py`, which has a `--help` command for usage information.
