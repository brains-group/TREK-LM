from datasets import load_dataset
import json
import os
from math import comb
import random
from transformers import AutoTokenizer

random.seed(1)

USER_STRING = "user"
ASSISTANT_STRING = "assistant"
SYSTEM_STRING = "system"

CONTENT_STRING = "content"
ROLE_STRING = "role"

HEAD_STRING = "head"
RELATION_STRING = "relation"
TAIL_STRING = "tail"

LIKED_STRING = "liked"
DISLIKED_STRING = "disliked"
SEEN_STRING = "seen"
UNSEEN_STRING = "unseen"
SUGGESTED_STRING = "suggested"

PROMPT_STRING = "prompt"
COMPLETION_STRING = "completion"
LABEL_STRING = "label"
GOAL_STRING = "goal"

KG_PREFACE_STRING = "You perform Knowledge Graph Completion. You will recommend a new triple to add to the user's knowledge graph with a tail entity that isn't already in their knowledge graph. The user's entity is represented by {}. Use this knowledge graph when responding to their queries: {}"
REQUEST_STRING = "Recommend movies to me."
COMPLETION_FORMAT_STRING = "Based on your Knowledge Graph, I recommend the following:\n{}Here is/are the triple(s) I think you should add to your knowledge graph:\n{}"


def triplesToStructured(triples):
    structured = {}
    for tripleIndex in range(len(triples[HEAD_STRING])):
        head = triples[HEAD_STRING][tripleIndex]
        relation = triples[RELATION_STRING][tripleIndex]
        tail = triples[TAIL_STRING][tripleIndex]
        if head not in structured:
            structured[head] = {}
        if relation not in structured[head]:
            structured[head][relation] = tail
        elif isinstance(structured[head][relation], list):
            structured[head][relation].append(tail)
        else:
            structured[head][relation] = [structured[head][relation], tail]
    return structured


def prefaceTurn(user, kg):
    return KG_PREFACE_STRING.format(user, json.dumps(triplesToStructured(kg)))


movieDataset = load_dataset("community-datasets/re_dial")

dataset = movieDataset["train"].to_dict()
testDataset = movieDataset["test"].to_dict()
for key in dataset.keys():
    dataset[key].extend(testDataset[key])

knowledgeGraphs = {}
kgDataset = {}
for index in range(len(dataset["conversationId"])):
    movieMentions = {
        movie["movieId"]: movie["movieName"]
        for movie in dataset["movieMentions"][index]
    }
    userId = dataset["initiatorWorkerId"][index]
    messages = dataset["messages"][index]

    questions = {
        movie["movieId"]: movie
        for movie in (
            dataset["respondentQuestions"][index]
            if dataset["initiatorQuestions"][index] == []
            else dataset["initiatorQuestions"][index]
        )
    }
    if questions == {}:
        continue

    if userId not in knowledgeGraphs:
        knowledgeGraphs[userId] = {
            HEAD_STRING: [],
            RELATION_STRING: [],
            TAIL_STRING: [],
        }
    userKG = knowledgeGraphs[userId]

    prompt = [
        {
            CONTENT_STRING: prefaceTurn(userId, userKG),
            ROLE_STRING: SYSTEM_STRING,
        }
    ]

    def updateUserKG(movieName, question, isAssistantMessage=False):
        try:
            index = userKG[TAIL_STRING].index(movieName)
        except:
            userKG[HEAD_STRING].append(userId)
            userKG[TAIL_STRING].append(movieName)
            userKG[RELATION_STRING].append("")
            index = len(userKG[RELATION_STRING]) - 1

        if question[LIKED_STRING] != 2:
            userKG[RELATION_STRING][index] = (
                LIKED_STRING if question[LIKED_STRING] == 1 else DISLIKED_STRING
            )
        elif question[SEEN_STRING] != 2:
            if userKG[RELATION_STRING][index] not in [LIKED_STRING, DISLIKED_STRING]:
                userKG[RELATION_STRING][index] = (
                    SEEN_STRING if question[SEEN_STRING] == 1 else UNSEEN_STRING
                )
        elif question[SUGGESTED_STRING] == 1 and isAssistantMessage:
            if userKG[RELATION_STRING][index] not in [
                LIKED_STRING,
                DISLIKED_STRING,
                SEEN_STRING,
                UNSEEN_STRING,
            ]:
                userKG[RELATION_STRING][index] = SUGGESTED_STRING
        elif userKG[RELATION_STRING] == "":
            del userKG[HEAD_STRING][-1]
            del userKG[TAIL_STRING][-1]
            del userKG[RELATION_STRING][-1]

    for message in messages:
        turn = {
            CONTENT_STRING: message["text"],
            ROLE_STRING: (
                USER_STRING if message["senderWorkerId"] == userId else ASSISTANT_STRING
            ),
        }

        moviesAdded = []
        for movieId, movieName in movieMentions.items():
            if movieName == None:
                continue
            newContent = turn[CONTENT_STRING].replace(f"@{movieId}", movieName)

            if newContent != turn[CONTENT_STRING] and movieId in questions:
                if turn[ROLE_STRING] == USER_STRING:
                    updateUserKG(movieName, questions[movieId])
                else:
                    moviesAdded.append((movieId, movieName))

            turn[CONTENT_STRING] = newContent

        if turn[ROLE_STRING] == ASSISTANT_STRING and len(moviesAdded) > 0:
            if userId not in kgDataset:
                kgDataset[userId] = []

            truncatedPrompt = prompt[
                : next(
                    (
                        turnIndex
                        for turnIndex in range(len(prompt), 0, -1)
                        if prompt[turnIndex - 1][ROLE_STRING] == USER_STRING
                    ),
                    0,
                )
            ]
            if len(truncatedPrompt) == 0:
                break

            goodMovies = ""
            goodTriples = ""
            goals = []
            badMovies = ""
            badTriples = ""
            for movieId, movieName in moviesAdded:
                newTriple = {userId: {SUGGESTED_STRING: movieName}}
                isGood = (
                    movieName not in userKG[TAIL_STRING]
                    and questions[movieId][LIKED_STRING] != 0
                )
                if isGood:
                    goodMovies += "- " + movieName + "\n"
                    goodTriples += json.dumps(newTriple) + "\n"
                    goal = movieName
                    if goal[-1] == ")":
                        goal = goal[: goal.rfind(" ")]
                    goals.append(goal)
                else:
                    badMovies += "- " + movieName + "\n"
                    badTriples += json.dumps(newTriple) + "\n"

                updateUserKG(movieName, questions[movieId], True)

            def addDataPoint(movies, triples, label, goals=[]):
                kgDataset[userId].append(
                    {
                        PROMPT_STRING: truncatedPrompt,
                        COMPLETION_STRING: [
                            {
                                CONTENT_STRING: COMPLETION_FORMAT_STRING.format(
                                    movies,
                                    triples,
                                ),
                                ROLE_STRING: ASSISTANT_STRING,
                            }
                        ],
                        LABEL_STRING: label,
                        GOAL_STRING: goals,
                    }
                )

            if len(goodMovies) > 0:
                addDataPoint(goodMovies, goodTriples, True, goals)
            if len(badMovies) > 0:
                addDataPoint(badMovies, badTriples, False)

        prompt.append(turn)
        prompt[0][CONTENT_STRING] = prefaceTurn(userId, userKG)


testProportion = 1 / 10
syntheticTestProportion = 1 / 3
realBenchmarkDataset = []
syntheticBenchmarkDataset = []

sumDataPoints = 0
sumPositiveDataPoints = 0
sumNegativeDataPoints = 0
culledKGDataset = {}
culledDataPoints = 0
culledUsers = 0
for user in kgDataset.keys():
    # remove items for test datasets
    removedItems = 0
    for index in range(len(kgDataset[user])):
        if kgDataset[user][index - removedItems]["label"]:
            if random.random() < testProportion:
                realBenchmarkDataset.append(kgDataset[user][index - removedItems])
                del kgDataset[user][index - removedItems]
                removedItems += 1

    sumChoices = sum([entry["label"] for entry in kgDataset[user]])
    numDataPoints = len(kgDataset[user])
    if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
        culledUsers += 1
        culledDataPoints += numDataPoints
    else:
        culledKGDataset[user] = kgDataset[user]
        sumDataPoints += numDataPoints
        sumPositiveDataPoints += sumChoices
        sumNegativeDataPoints += numDataPoints - sumChoices


print("--------- Base KG Dataset ---------")
print("Number of users: " + str(len(culledKGDataset.keys())))
print("Number of data points: " + str(sumDataPoints))
print("Number of positive data points: " + str(sumPositiveDataPoints))
print("Number of negative data points: " + str(sumNegativeDataPoints))
print(
    "Positive to Negative Ratio: " + str(sumPositiveDataPoints / sumNegativeDataPoints)
)
print("Number of culled users: " + str(culledUsers))
print("Number of culled datapoints: " + str(culledDataPoints))

with open("movieKnowledgeGraphDataset.json", "w") as file:
    json.dump(culledKGDataset, file, indent=4)


nonFederatedDataset = []
for user in culledKGDataset.values():
    nonFederatedDataset.extend(user)

with open("nonFederatedMovieKnowledgeGraphDataset.json", "w") as file:
    json.dump(nonFederatedDataset, file, indent=4)


# Synthetic Data
syntheticStartIndexes = {}
for user in kgDataset.keys():
    syntheticStartIndexes[user] = len(knowledgeGraphs[user][HEAD_STRING])
    goodChoices = [
        index
        for index in range(len(knowledgeGraphs[user][RELATION_STRING]))
        if knowledgeGraphs[user][RELATION_STRING][index] == LIKED_STRING
    ]
    badChoices = [
        index
        for index in range(len(knowledgeGraphs[user][RELATION_STRING]))
        if knowledgeGraphs[user][RELATION_STRING][index] == DISLIKED_STRING
    ]
    repChoices = list(range(len(knowledgeGraphs[user][RELATION_STRING])))

    userKG = knowledgeGraphs[user]

    def generateSyntheticCompletion(choiceOptions, label, shouldDelete=True):
        choiceIndexes = random.sample(
            choiceOptions, random.randint(1, min(10, len(choiceOptions)))
        )
        specificKG = {
            HEAD_STRING: userKG[HEAD_STRING].copy(),
            RELATION_STRING: userKG[RELATION_STRING].copy(),
            TAIL_STRING: userKG[TAIL_STRING].copy(),
        }

        movies = ""
        triples = ""
        goals = []
        for choiceIndex in choiceIndexes:
            newTriple = {
                userKG[HEAD_STRING][choiceIndex]: {
                    SUGGESTED_STRING: userKG[TAIL_STRING][choiceIndex]
                }
            }

            if label:
                movies += "- " + userKG[TAIL_STRING][choiceIndex] + "\n"
                triples += json.dumps(newTriple) + "\n"
                goal = userKG[TAIL_STRING][choiceIndex]
                if goal[-1] == ")":
                    goal = goal[: goal.rfind(" ")]
                goals.append(goal)
            else:
                movies += "- " + userKG[TAIL_STRING][choiceIndex] + "\n"
                triples += json.dumps(newTriple) + "\n"
        choiceIndexes.sort(reverse=True)
        for choiceIndex in choiceIndexes:
            if shouldDelete:
                del specificKG[HEAD_STRING][choiceIndex]
                del specificKG[RELATION_STRING][choiceIndex]
                del specificKG[TAIL_STRING][choiceIndex]

        completion = [
            {
                CONTENT_STRING: COMPLETION_FORMAT_STRING.format(
                    movies,
                    triples,
                ),
                ROLE_STRING: ASSISTANT_STRING,
            }
        ]

        newDatapoint = {
            PROMPT_STRING: [
                {
                    CONTENT_STRING: prefaceTurn(user, specificKG),
                    ROLE_STRING: SYSTEM_STRING,
                },
                {
                    CONTENT_STRING: REQUEST_STRING,
                    ROLE_STRING: USER_STRING,
                },
            ],
            COMPLETION_STRING: completion,
            LABEL_STRING: label,
            GOAL_STRING: goals,
        }
        # sometimes add to test dataset instead of normal dataset
        if label and random.random() < syntheticTestProportion:
            syntheticBenchmarkDataset.append(newDatapoint)
        else:
            kgDataset[user].append(newDatapoint)

    if len(goodChoices) > 0:
        for i in range(random.randint(0, len(goodChoices))):
            generateSyntheticCompletion(goodChoices, True)
    if len(badChoices) > 0:
        for i in range(random.randint(0, len(badChoices))):
            generateSyntheticCompletion(badChoices, False)
    if len(repChoices) > 0:
        for i in range(random.randint(0, len(repChoices))):
            generateSyntheticCompletion(repChoices, False, False)

sumDataPoints = 0
sumPositiveDataPoints = 0
sumNegativeDataPoints = 0
culledKGDataset = {}
culledDataPoints = 0
culledUsers = 0
for user in kgDataset.keys():
    sumChoices = sum([entry["label"] for entry in kgDataset[user]])
    numDataPoints = len(kgDataset[user])
    if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
        culledUsers += 1
        culledDataPoints += numDataPoints
        if syntheticStartIndexes[user] > 0:
            realBenchmarkDataset.extend(
                [
                    entry
                    for entry in kgDataset[user][: syntheticStartIndexes[user]]
                    if entry["label"]
                ]
            )
        if syntheticStartIndexes[user] < len(kgDataset[user]):
            syntheticBenchmarkDataset.extend(
                [
                    entry
                    for entry in kgDataset[user][syntheticStartIndexes[user] :]
                    if entry["label"]
                ]
            )
    else:
        culledKGDataset[user] = kgDataset[user]
        sumDataPoints += numDataPoints
        sumPositiveDataPoints += sumChoices
        sumNegativeDataPoints += numDataPoints - sumChoices


print("--------- KG Dataset with Synthetic Data ---------")
print("Number of users: " + str(len(culledKGDataset.keys())))
print("Number of data points: " + str(sumDataPoints))
print("Number of positive data points: " + str(sumPositiveDataPoints))
print("Number of negative data points: " + str(sumNegativeDataPoints))
print(
    "Positive to Negative Ratio: " + str(sumPositiveDataPoints / sumNegativeDataPoints)
)
print("Number of culled users: " + str(culledUsers))
print("Number of culled datapoints: " + str(culledDataPoints))

with open("movieKnowledgeGraphDatasetWithSyntheticData.json", "w") as file:
    json.dump(culledKGDataset, file, indent=4)


nonFederatedSyntheticDataset = []
for user in culledKGDataset.values():
    nonFederatedSyntheticDataset.extend(user)


with open("nonFederatedMovieKnowledgeGraphDatasetWithSyntheticData.json", "w") as file:
    json.dump(nonFederatedSyntheticDataset, file, indent=4)


numPositiveTest = sum([entry["label"] for entry in realBenchmarkDataset])
print("--------- KG Test Dataset ---------")
print("Number of data points: " + str(len(realBenchmarkDataset)))

with open("movieKnowledgeGraphTestDataset.json", "w") as file:
    json.dump(realBenchmarkDataset, file, indent=4)


numPositiveTest = sum([entry["label"] for entry in syntheticBenchmarkDataset])
print("--------- KG Synthetic Test Dataset ---------")
print("Number of data points: " + str(len(syntheticBenchmarkDataset)))

with open("movieKnowledgeGraphSyntheticTestDataset.json", "w") as file:
    json.dump(syntheticBenchmarkDataset, file, indent=4)


entities = set()
relations = set()
test = []
testProportion = (len(realBenchmarkDataset) + len(syntheticBenchmarkDataset)) / len(
    nonFederatedSyntheticDataset
)
valid = []
validProportion = 17535 / 272116
train = []
for kg in knowledgeGraphs.values():
    entities.update(kg[HEAD_STRING])
    entities.update(kg[TAIL_STRING])
    relations.update(kg[RELATION_STRING])

    valid.append([])
    train.append([])

    kgTriples = [
        f"{str(head).replace(" ", "_")}\t{relation.replace(" ", "_")}\t{str(tail).replace(" ", "_")}\n"
        for head, relation, tail in zip(
            kg[HEAD_STRING], kg[RELATION_STRING], kg[TAIL_STRING]
        )
    ]
    for triple in kgTriples:
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

path = "../{}/data/movieKnowledgeGraphDataset"
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
with open(f"{pathHAKE}/entities.txt", "w") as file:
    file.writelines([f"{index}\t{entity}\n" for index, entity in enumerate(entities)])
with open(f"{pathHAKE}/relations.txt", "w") as file:
    file.writelines(
        [f"{index}\t{relation}\n" for index, relation in enumerate(relations)]
    )

pathKBGAT = path.format("KBGAT")
with open(f"{pathKBGAT}/entity2id.txt", "w") as file:
    file.writelines([f"{entity}\t{index}\n" for index, entity in enumerate(entities)])
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
