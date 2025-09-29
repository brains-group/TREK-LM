import json
import os
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm


def find_longest_tokenized_prompt(dataset, tokenizer, dataset_name):
    longest_prompt_text = ""
    max_token_length = 0

    print(f"\nAnalyzing {dataset_name} for longest tokenized prompt...")

    # Iterate through users if it's a federated dataset (dict of lists)
    if isinstance(dataset, dict):
        all_data_points = [item for sublist in dataset.values() for item in sublist]
    # Otherwise, assume it's a non-federated or benchmark dataset (list of dicts)
    else:
        all_data_points = dataset

    for data_point in tqdm(
        all_data_points, desc=f"Tokenizing prompts in {dataset_name}"
    ):
        prompt_messages = data_point["prompt"]
        current_prompt_text = " ".join(
            [msg["content"] for msg in prompt_messages]
        )

        # Only tokenize if the current prompt is longer in characters than the previous longest
        if len(current_prompt_text) > len(longest_prompt_text):
            tokenized_length = len(tokenizer.encode(current_prompt_text))
            if tokenized_length > max_token_length:
                max_token_length = tokenized_length
                longest_prompt_text = current_prompt_text

    if max_token_length > 0:
        print(
            f"Longest tokenized prompt in {dataset_name} (tokens): {max_token_length}"
        )
        print(f"Longest prompt text (first 200 chars): {longest_prompt_text[:200]}...")
    else:
        print(f"No prompts found in {dataset_name}.")
    return max_token_length


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
