from itertools import chain
import json
import os
import random
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
        current_prompt_text = " ".join([msg["content"] for msg in prompt_messages])

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


def adapt_HAKE_and_KBGAT_and_FedKGRec_data(
    knowledge_graphs,
    testProportion,
    validProportion,
    folderName,
    tripleTransform,
    tripleTest,
):
    entities = set()
    relations = set()
    test = []
    valid = []
    train = []
    for kg in tqdm(knowledge_graphs):
        if not kg:
            continue
        valid.append([])
        train.append([])
        for s, p, o in kg.triples((None, None, None)):
            entities.add(str(s).replace(" ", "_"))
            entities.add(str(o).replace(" ", "_"))
            relations.add(p.replace(" ", "_"))

            triple = f"{str(s).replace(" ", "_")}\t{p.replace(" ", "_")}\t{str(o).replace(" ", "_")}\n"
            if random.random() < testProportion:
                test.append(triple)
            elif random.random() < validProportion:
                valid[-1].append(triple)
            else:
                train[-1].append(triple)

        numDataPoints = len(valid[-1]) + len(train[-1])
        if numDataPoints < 10:
            test.extend(train[-1])
            del train[-1]
            test.extend(valid[-1])
            del valid[-1]

    nonFederatedValid = [triple for triples in valid for triple in triples]
    nonFederatedTrain = [triple for triples in train for triple in triples]

    path = "../{}/data/" + folderName
    for modelName in ["HAKE", "KBGAT"]:
        loopPath = path.format(modelName)
        os.makedirs(loopPath, exist_ok=True)
        with open(f"{loopPath}/train.txt", "w") as file:
            file.writelines(nonFederatedTrain)
        with open(f"{loopPath}/test.txt", "w") as file:
            file.writelines(test)
        with open(f"{loopPath}/valid.txt", "w") as file:
            file.writelines(nonFederatedValid)

        federatedPath = loopPath + "/federated"
        os.makedirs(federatedPath, exist_ok=True)
        for index in range(len(train)):
            with open(f"{federatedPath}/train{index}.txt", "w") as file:
                file.writelines(train[index])
            with open(f"{federatedPath}/valid{index}.txt", "w") as file:
                file.writelines(valid[index])

    pathHAKE = path.format("HAKE")
    with open(f"{pathHAKE}/entities.dict", "w") as file:
        file.writelines(
            [f"{index}\t{entity}\n" for index, entity in enumerate(entities)]
        )
    with open(f"{pathHAKE}/relations.dict", "w") as file:
        file.writelines(
            [f"{index}\t{relation}\n" for index, relation in enumerate(relations)]
        )

    pathKBGAT = path.format("KBGAT")
    with open(f"{pathKBGAT}/entity2id.txt", "w") as file:
        file.writelines(
            [f"{entity}\t{index}\n" for index, entity in enumerate(entities)]
        )
    with open(f"{pathKBGAT}/relation2id.txt", "w") as file:
        file.writelines(
            [f"{relation}\t{index}\n" for index, relation in enumerate(relations)]
        )
    with open(f"{pathKBGAT}/entity2vec.txt", "w") as file:
        file.writelines(
            [
                ("\t".join([str(random.random() - 0.5) for _ in range(100)]) + "\n")
                for entity in entities
            ]
        )
    with open(f"{pathKBGAT}/relation2vec.txt", "w") as file:
        file.writelines(
            [
                ("\t".join([str(random.random() - 0.5) for _ in range(100)]) + "\n")
                for relation in relations
            ]
        )

    pathFedKGRec = path.format("FedKGRec")
    os.makedirs(pathFedKGRec, exist_ok=True)
    entitiesList = list(entities)
    with open(f"{pathFedKGRec}/item_index2entity_id.txt", "w") as file:
        file.writelines(
            [f"{index}::{entity}\n" for index, entity in enumerate(entitiesList)]
        )
    with open(f"{pathFedKGRec}/kg.txt", "w") as file:
        file.writelines(
            [
                triple.replace("\t", "::")
                for triple in chain(nonFederatedTrain, test, nonFederatedValid)
                if not tripleTest(triple)
            ]
        )

    def tripleTransformFedKGRec(triple):
        user, rating, item = triple.split("\t")
        item = item.replace("\n", "")
        return f"{entitiesList.index(user)}::{entitiesList.index(item)}::{rating}\n"

    with open(f"{pathFedKGRec}/ratings.dat", "w") as file:
        file.writelines(
            [
                tripleTransformFedKGRec(tripleTransform(triple))
                for triple in chain(nonFederatedTrain, test, nonFederatedValid)
                if tripleTest(triple)
            ]
        )
