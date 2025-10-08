# Personalized Knowledge Graph Completion Using Federated Large Language Models

## Requirements

The requirements.txt file contains a simple list of required libraries. The required dependencies will be handled by pip. Run the following to install the dependencies:

    pip install -r requirements.txt

## Generating the Data

The movie KG dataset can be generated simply by running `createMovieKGData.py` in the `data` folder. This will also generate the adapted datasets for KBGAT and HAKE.


For the [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset, the code is a little more complicated.
If you have conda, you can just run `run_convertFoodIngrMap_in_env.sh` in the `TREK-LM/data/FoodIngrMapWorkaround` folder. Then, you can run `createRecipeKGDataset.py` to complete the data generation.

## Training the Models

### LLM Model

The LLM model can be trained normally by running `train_centralized.py`. The dataset, model, and hyperparameters can be adjusted by editing the yaml files in the `conf` folder.
Additionally, you can override the base model path by passing a `--base_model_path` argument. You can also override the dataset name used by passing the `--dataset_name` argument, and you can pass the `--dataset_index` if you are using a federated dataset and want to train on just one client's data (used for the local training ablation).

The LLM model can be trained via a federated simulation by running `train.py`. The dataset, model, and hyperparameters can be adjusted by editing the yaml files in the `conf` folder.
Additionally, you can override the base model path by passing a `--base_model_path` argument, and you can override the number of federated training rounds by passing the `--num_rounds` argument (useful for resuming from a checkpoint). You can also override the dataset name used by passing the `--dataset_name` argument.

### KBGAT

The KBGAT model can be trained normally by running `main.py` in  the `KBGAT` folder using the recommended settings from the original [repo](https://github.com/deepakn97/relationPrediction/tree/master).

It can be trained via a federated simulation by running `train_fed_KBGAT.py` in the `KBGAT` folder.

### HAKE

The HAKE model can be trained normally by running `runs.py` in the `HAKE/codes` folder using the recommended settings from the original [repo](https://github.com/MIRALab-USTC/KGE-HAKE/tree/master).

It can be trained via a federated simulation by running `train_fed_HAKE.py` in the `HAKE/codes` folder.

## Testing the models

### LLM Model

The LLM model can be tested by running `test.py` and passing the path to the LoRA checkpoint via the `--lora_path` argument, and you can use a different base model by passing the `--base_model_path` argument (defaults to Qwen/Qwen3-0.6B). You can also pass the `--userID` argument to test on datapoints based on exclusively one user's data.

### KBGAT

The KBGAT model can be tested by running `test_KBGAT.py` in the `KBGAT` folder.

### HAKE

The HAKE model can be tested by running `test_HAKE.py` in the `HAKE/codes` folder.

## Local Model Training Ablation Tests

The local model training ablation test for the federated model can be done simply by running `testAnalysis.py`, which will handle the training and testing all at once.

## External Datasets

The external dataset used in this repo is [ReDial](https://huggingface.co/datasets/community-datasets/re_dial), but it doesn't need to be manually downloaded. The `createMovieKGData.py` code automatically pulls ReDial from HuggingFace.

For the [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset, it is also automatically downloaded by the code.
