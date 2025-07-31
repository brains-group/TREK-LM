# Personalized Knowledge Graph Completion Using Federated Large Language Models

## Requirements

The requirements.txt file contains a simple list of required libraries. The required dependencies will be handled by pip. Run the following to install the dependencies:

    pip install -r requirements.txt

## Generating the Data

The movie KG dataset can be generated simply by running `createMovieData.py` in the `data` folder. This will also generate the adapted datasets for KBGAT and HAKE.

## Training the Models

### LLM Model

The LLM model can be trained normally by running `centralized_train.py`. The dataset, model, and hyperparameters can be adjusted by editting `centralized_full.yaml` in the `conf` folder.

The LLM model can be trained via a federated simulation by running `train.py`. The dataset, model, and hyperparameters can be adjusted by editting `federated_full.yaml` in the `conf` folder.

### KBGAT

The KBGAT model can be trained normally by running `main.py` in  the `KBGAT` folder using the recommended settings from the original [repo](https://github.com/deepakn97/relationPrediction/tree/master).

It can be trained via a federated simulation by running `train_fed_KBGAT.py` in the `KBGAT` folder.

### HAKE

The HAKE model can be trained normally by running `runs.py` in the `HAKE/codes` folder using the recommended settings from the original [repo](https://github.com/MIRALab-USTC/KGE-HAKE/tree/master).

It can be trained via a federated simulation by running `train_fed_HAKE.py` in the `HAKE/codes` folder.

## Testing the models

### LLM Model

The LLM model can be tested by running `test.py` and passing the path to the LoRA checkpoint via the `--lora_path` argument.

### KBGAT

The KBGAT model can be tested by running `test_KBGAT.py` in the `KBGAT` folder.

### HAKE

The HAKE model can be tested by running `test_HAKE.py` in the `HAKE/codes` folder.

## External Datasets

The external dataset used in this repo is [ReDial](https://huggingface.co/datasets/community-datasets/re_dial), but it doesn't need to be manually downloaded. The `createMovieData.py` code automatically pulls ReDial from HuggingFace.
