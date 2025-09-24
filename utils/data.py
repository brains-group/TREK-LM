import json
import os
import pandas as pd
from datasets import load_dataset, Dataset


def format_dataset(dataset):
    """Removes and renames columns to match the expected format."""
    dataset = dataset.remove_columns(["instruction"])
    dataset = dataset.rename_column("output", "response")
    dataset = dataset.rename_column("input", "instruction")
    return dataset


def formatting_prompts_func(example):
    """Formats a batch of examples into Alpaca-style prompts."""
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def load_jsonl(filename):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
    return data


def aggregate_datasets(path, subsets, partition="train"):
    """
    Takes as input a Huggingface DatasetDict with subset name as key, and Dataset as value.
    Returns a pd.DataFrame with all subsets concatenated.

    :param subsets: list of str, the subsets of the data to download from the HuggingFace hub.
    :return: pd.DataFrame
    """
    dataframes = []
    for subset in subsets:
        subset_data = load_dataset(os.path.join(path, subset), split=partition)
        subset_df = pd.DataFrame(subset_data.map(lambda x: {"subset": subset, **x}))
        dataframes.append(subset_df)
    aggregate_df = pd.concat(dataframes, axis=0)
    aggregate = Dataset.from_pandas(aggregate_df)
    if "__index_level_0__" in aggregate.column_names:
        aggregate = aggregate.remove_columns("__index_level_0__")
    return aggregate


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
