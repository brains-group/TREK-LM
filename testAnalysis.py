import json
import os
import re
from collections import defaultdict
import random
import subprocess
import sys

from tqdm import tqdm

random.seed(2025)

with open("./data/movieKnowledgeGraphTestDataset.json", "r") as file:
    dataset = json.load(file)

userCounts = defaultdict(int)
for datapoint in dataset:
    user = re.search(
        r"The user's entity is represented by (\d+).", datapoint["prompt"][0]["content"]
    ).group(1)
    userCounts[user] += 1

selectedUsers = random.sample(
    list(filter(lambda pair: pair[1] >= 10, userCounts.items())), 10
)
print(selectedUsers)

realPrecision = 0
realRecall = 0
realMrr = 0
realHits1 = 0
realHits3 = 0
realHits10 = 0
syntheticPrecision = 0
syntheticRecall = 0
syntheticMrr = 0
syntheticHits1 = 0
syntheticHits3 = 0
syntheticHits10 = 0
for selectedUser in tqdm(selectedUsers):
    # print(f"Performing Training for User: {selectedUser}")
    # command = [
    #     sys.executable,
    #     "centralized_train.py",
    #     "--dataset_name",
    #     "movieKnowledgeGraphDataset",
    #     "--dataset_index",
    #     selectedUser[0],
    # ]
    # subprocess.run(command, check=True)

    print(f"Performing Test for User: {selectedUser}", flush=True)
    loraPath = f"./models/centralized/Qwen/Qwen3-0.6B/movieKnowledgeGraphDataset/{selectedUser[0]}/"
    loraPath = os.path.join(loraPath, max(os.listdir(loraPath)))
    command = [
        sys.executable,
        "test.py",
        "--lora_path",
        loraPath,
        "--userID",
        selectedUser[0],
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout
    print(output, flush=True)

    scores = output[output.rfind("Real Data Scores: ") :]
    realPrecision += float(re.search(r"Precision: (\d*\.?\d*)", scores).group(1))
    realRecall += float(re.search(r"Recall: (\d*\.?\d*)", scores).group(1))
    realMrr += float(re.search(r"MRR: (\d*\.?\d*)", scores).group(1))
    realHits1 += float(re.search(r"Hits@1: (\d*\.?\d*)", scores).group(1))
    realHits3 += float(re.search(r"Hits@3: (\d*\.?\d*)", scores).group(1))
    realHits10 += float(re.search(r"Hits@10: (\d*\.?\d*)", scores).group(1))

    # print(f"Performing Synthetic Training for User: {selectedUser}")
    # command = [
    #     sys.executable,
    #     "centralized_train.py",
    #     "--dataset_name",
    #     "movieKnowledgeGraphDatasetWithSyntheticData",
    #     "--dataset_index",
    #     selectedUser[0],
    # ]
    # subprocess.run(command, check=True)

    print(f"Performing Test for Synthetic User: {selectedUser}", flush=True)
    loraPath = f"./models/centralized/Qwen/Qwen3-0.6B/movieKnowledgeGraphDatasetWithSyntheticData/{selectedUser[0]}/"
    loraPath = os.path.join(loraPath, max(os.listdir(loraPath)))
    command = [
        sys.executable,
        "test.py",
        "--lora_path",
        loraPath,
        "--userID",
        selectedUser[0],
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout
    print(output, flush=True)

    scores = output[output.rfind("Real Data Scores: ") :]
    syntheticPrecision += float(re.search(r"Precision: (\d*\.?\d*)", scores).group(1))
    syntheticRecall += float(re.search(r"Recall: (\d*\.?\d*)", scores).group(1))
    syntheticMrr += float(re.search(r"MRR: (\d*\.?\d*)", scores).group(1))
    syntheticHits1 += float(re.search(r"Hits@1: (\d*\.?\d*)", scores).group(1))
    syntheticHits3 += float(re.search(r"Hits@3: (\d*\.?\d*)", scores).group(1))
    syntheticHits10 += float(re.search(r"Hits@10: (\d*\.?\d*)", scores).group(1))

numSelectedUsers = len(selectedUsers)
print(
    "Real Results: \nNumber of Selected Users: {}\nPrecision: {}\nRecall: {}\nMRR: {}\nHits@1: {}\nHits@3: {}\nHits@10: {}".format(
        numSelectedUsers,
        realPrecision / numSelectedUsers,
        realRecall / numSelectedUsers,
        realMrr / numSelectedUsers,
        realHits1 / numSelectedUsers,
        realHits3 / numSelectedUsers,
        realHits10 / numSelectedUsers,
    ),
    flush=True,
)
print(
    "Synthetic Results: \nNumber of Selected Users: {}\nPrecision: {}\nRecall: {}\nMRR: {}\nHits@1: {}\nHits@3: {}\nHits@10: {}".format(
        numSelectedUsers,
        syntheticPrecision / numSelectedUsers,
        syntheticRecall / numSelectedUsers,
        syntheticMrr / numSelectedUsers,
        syntheticHits1 / numSelectedUsers,
        syntheticHits3 / numSelectedUsers,
        syntheticHits10 / numSelectedUsers,
    ),
    flush=True,
)
