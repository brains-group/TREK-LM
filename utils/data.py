import json
import os
import pandas as pd
from datasets import load_dataset, Dataset


def load_centralized_dataset(path: str, index: str = None):
    """
    Loads a dataset for centralized training from a JSON file.

    :param path: str, the path to the JSON file.
    :param index: str (optional), the key to extract from the JSON file if it's a dictionary.
    :return: HuggingFace Dataset
    """
    with open(path, "r") as file:
        json_data = json.load(file)
        if index:
            json_data = json_data[index]
    return Dataset.from_list(json_data)


def load_federated_dataset(path: str):
    """
    Loads a dataset for federated training from a JSON file.
    The file is expected to be a dictionary where keys are client IDs.

    :param path: str, the path to the JSON file.
    :return: dict
    """
    with open(path, "r") as file:
        datasets = json.load(file)
    return datasets
