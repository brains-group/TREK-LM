import math
import random
import json
from tqdm import tqdm

random.seed(1)

USER_STRING = "user"
ASSISTANT_STRING = "assistant"
SYSTEM_STRING = "system"

CONTENT_STRING = "content"
ROLE_STRING = "role"

PROMPT_STRING = "prompt"
COMPLETION_STRING = "completion"
LABEL_STRING = "label"
GOAL_STRING = "goal"

PREFACE_STRING = "You perform Knowledge Graph Completion. The user will provide a triple with a missing head or tail, and you will suggest completions of that triple."
REQUEST_STRING = "Suggest completions for this triple: {}"
COMPLETION_FORMAT_STRING = "I suggest the following completions:\n{}"


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return (e1, relation, e2)


with open("./train.txt") as f:
    trainLines = f.readlines()
with open("./valid.txt") as f:
    trainLines += f.readlines()
with open("./test.txt") as f:
    testLines = f.readlines()


def linesToGroupedTriples(lines):
    groupedTriples = {}
    for line in tqdm(lines):
        triple = parse_line(line)
        headEmpty = (None, triple[1], triple[2])
        if headEmpty not in groupedTriples:
            groupedTriples[headEmpty] = []
        groupedTriples[headEmpty].append(triple[0])
        tailEmpty = (triple[0], triple[1], None)
        if tailEmpty not in groupedTriples:
            groupedTriples[tailEmpty] = []
        groupedTriples[tailEmpty].append(triple[2])
    return groupedTriples


trainGroupedTriples = linesToGroupedTriples(trainLines)
testGroupedTriples = linesToGroupedTriples(testLines)

for key in trainGroupedTriples.keys():
    if key in testGroupedTriples:
        testGroupedTriples[key].extend(trainGroupedTriples[key])


def createCompletionString(triple, completions, label=True):
    return {
        CONTENT_STRING: (COMPLETION_FORMAT_STRING if label else "{}").format(
            ("\n" if label else " ").join(
                [
                    f"{"- " if label else ""}{" -> ".join([completion if val is None else val for val in triple])}"
                    for completion in completions
                ]
            )
        ),
        ROLE_STRING: ASSISTANT_STRING,
    }


def groupedTriplesToDataset(groupedTriples, includeGoal=False):
    dataset = []
    for triple, completions in tqdm(groupedTriples.items()):
        dataset.append(
            {
                PROMPT_STRING: [
                    {
                        CONTENT_STRING: PREFACE_STRING,
                        ROLE_STRING: SYSTEM_STRING,
                    },
                    {
                        CONTENT_STRING: REQUEST_STRING.format(
                            " -> ".join(
                                ["BLANK" if val is None else val for val in triple]
                            )
                        ),
                        ROLE_STRING: USER_STRING,
                    },
                ],
                COMPLETION_STRING: createCompletionString(triple, completions),
                LABEL_STRING: True,
            }
        )
        if includeGoal:
            dataset[-1][GOAL_STRING] = completions
        else:
            dataset.append(dataset[-1].copy())
            dataset[-1][LABEL_STRING] = False
            dataset[-1][COMPLETION_STRING] = createCompletionString(
                triple, random.choice(list(groupedTriples.values())), False
            )
    return dataset


nonFederatedDataset = groupedTriplesToDataset(trainGroupedTriples)
testDataset = groupedTriplesToDataset(testGroupedTriples, True)

partitionSize = math.ceil(len(nonFederatedDataset) / 20)
federatedDataset = [
    nonFederatedDataset[index : (index + partitionSize)]
    for index in range(0, len(nonFederatedDataset), partitionSize)
]

sumPositiveDataPoints = sum([datapoint["label"] for datapoint in nonFederatedDataset])
sumNegativeDataPoints = len(nonFederatedDataset) - sumPositiveDataPoints

print("--------- FB15k-237 Dataset ---------")
print("Number of partitions: " + str(len(federatedDataset)))
print("Number of data points: " + str(len(nonFederatedDataset)))
print("Number of positive data points: " + str(sumPositiveDataPoints))
print("Number of negative data points: " + str(sumNegativeDataPoints))
print(
    "Positive to Negative Ratio: " + str(sumPositiveDataPoints / sumNegativeDataPoints)
)

with open("FB15k-237.json", "w") as file:
    json.dump(nonFederatedDataset, file, indent=4)
with open("federatedFB15k-237.json", "w") as file:
    json.dump(federatedDataset, file, indent=4)

print("--------- FB15k-237 Test Dataset ---------")
print("Number of data points: " + str(len(testDataset)))

with open("testFB15k-237.json", "w") as file:
    json.dump(testDataset, file, indent=4)
