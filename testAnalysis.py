import json
import re
from collections import defaultdict
import random
import subprocess
import sys

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

for selectedUser in selectedUsers:
    print(f"Performing Training for User: {selectedUser}")
    command = [
        sys.executable,
        "centralized_train.py",
        "--dataset_name",
        "movieKnowledgeGraphDatasetWithSyntheticData",
        "--dataset_index",
        selectedUser[0],
    ]
    subprocess.run(command)
