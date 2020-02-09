# Code Prediction

## Summary
This project is an extension to 'Learning Python Code Suggestion with a Sparse Pointer Network' by [Bhoopchand et al.](https://arxiv.org/pdf/1611.08307.pdf) to share an attention between the copy mechanism and the vocabulary predictions.
Check out [Write-Up.pdf](Write-Up.pdf) for a detailed description of my work and experiments.

## Overview of the Repository
The code can be found in `src`, which is based on the language model provided in class.
It incorporates the model described by Bhoopchand et al. in `src/language_model`.
All code is compatible for Python version >=3.5.

## Dataset Preparation
You can use some directory of source code as dataset. You have to transform this dataset into a graph format for easier analysis with [this](https://github.com/acr31/features-javac/) tool. This can then be split up into subsets however you want and these can then be normalized with `src/dataset_normalization.py`. The needed dependencies for `src/dataset_normalziation.py` can be installed with `pip install -r src/normalization_requirements.txt`. Run `python3 src/dataset_normalizer.py --help` to see how to use the normalization script. This script writes to a file called `minimumleft.json` which can then be fed to it again in another run. `minimumleft.json` contains counts of how many tokens are unused for each token type and thus can be used in the next run to reduce the number of tokens to a minimum (+ padding).

### Ready Dataset
If you do not want to normalize your own data but just want to try the code on my split of `r252-corpus-features` you can download [here](https://drive.google.com/file/d/1J07bJP5dm36pmLNahBen1ZkVF8O1MLcq/view?usp=sharing). This dataset has a non-typed normalization and is split into `normalized_split/train`, `normalized_split/test` and `normalized_split/valid`. These directories can be used in the next step as train, test and validation data.

## Running the Language Model
You can run the `src/language_mode` code in the container provided in `src/container` that is available on Docker Hub as `samgmuller/tf1.12base:0.4`. If you use CodaLab you might find `src/coda.py` to be a helpful script for running experiments. If you do not want to use the container you can also install the `src/container/requirements.txt` as well as TensorFlow 1.12 . I recommend using a virtual environment to install TensorFlow.
### Training
You can run the training as described by `python3 src/language_model/train.py --help`. The easiest usage would be something like: `python3 src/language_model_train.py SAVE_MODEL_DIR DIR_WITH_TRAIN_DATA DIR_WITH_VALID_DATA`. The training and validation data is assumed to be in the proto format defined in `graph.proto`. 
### Evaluation
The evaluation can be run with `python3 src/language_model/evaluate.py`, which has a `--help` command for usage information.
### Plotting
To plot the results you might consider using the notebook `src/learning_curve_plotting.ipynb` that has some nice plots of the results and the learning curves found in `output_files`.
