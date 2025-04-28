from datasets import load_dataset
import json
import random

random.seed(1)

USER_STRING = "user"
ASSISTANT_STRING = "assistant"

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

KG_PREFACE_STRING = "My entity is represented by {}. Use this knowledge graph when responding to my queries: {}"
REQUEST_STRING = "Recommend a movie to me."


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
        if isinstance(structured[head][relation], list):
            structured[head][relation].append(tail)
        else:
            structured[head][relation] = [structured[head][relation], tail]
    return structured


def prefaceTurn(user, kg):
    return KG_PREFACE_STRING.format(userId, json.dumps(triplesToStructured(userKG)))


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
            ROLE_STRING: USER_STRING,
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

            for movieId, movieName in moviesAdded:
                kgDataset[userId].append(
                    {
                        PROMPT_STRING: prompt.copy(),
                        COMPLETION_STRING: [
                            {
                                CONTENT_STRING: json.dumps(
                                    {userId: {SUGGESTED_STRING: movieName}}
                                ),
                                ROLE_STRING: ASSISTANT_STRING,
                            }
                        ],
                        LABEL_STRING: movieName not in userKG[TAIL_STRING]
                        and questions[movieId][LIKED_STRING] != 0,
                    }
                )

                updateUserKG(movieName, questions[movieId], True)

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

with open("nonFederatedMovieKnowledgeGraphDataset.jsonl", "w") as file:
    for dataPoint in nonFederatedDataset:
        file.write(json.dumps(dataPoint) + '\n')


# Synthetic Data
syntheticStartIndexes = {}
for user in kgDataset.keys():
    syntheticStartIndexes[user] = len(knowledgeGraphs[user][HEAD_STRING])
    goodChoices = [
        index
        for index in range(len(knowledgeGraphs[user][RELATION_STRING]))
        if knowledgeGraphs[user][RELATION_STRING][index] == LIKED_STRING
    ]
    random.shuffle(goodChoices)
    badChoices = [
        index
        for index in range(len(knowledgeGraphs[user][RELATION_STRING]))
        if knowledgeGraphs[user][RELATION_STRING][index] == DISLIKED_STRING
    ]
    random.shuffle(badChoices)
    repChoices = list(range(len(knowledgeGraphs[user][RELATION_STRING])))
    random.shuffle(repChoices)

    userKG = knowledgeGraphs[user]

    def generateSyntheticCompletion(choiceIndex, label, shouldDelete=True):
        specificKG = {
            HEAD_STRING: userKG[HEAD_STRING].copy(),
            RELATION_STRING: userKG[RELATION_STRING].copy(),
            TAIL_STRING: userKG[TAIL_STRING].copy(),
        }
        completion = [
            {
                CONTENT_STRING: json.dumps(
                    {
                        userKG[HEAD_STRING][choiceIndex]: {
                            SUGGESTED_STRING: userKG[TAIL_STRING][choiceIndex]
                        }
                    }
                ),
                ROLE_STRING: ASSISTANT_STRING,
            }
        ]
        if shouldDelete:
            del specificKG[HEAD_STRING][choiceIndex]
            del specificKG[RELATION_STRING][choiceIndex]
            del specificKG[TAIL_STRING][choiceIndex]

        newDatapoint = {
            PROMPT_STRING: [
                {
                    CONTENT_STRING: prefaceTurn(user, specificKG),
                    ROLE_STRING: USER_STRING,
                },
                {
                    CONTENT_STRING: REQUEST_STRING,
                    ROLE_STRING: USER_STRING,
                },
            ],
            COMPLETION_STRING: completion,
            LABEL_STRING: label,
        }
        # sometimes add to test dataset instead of normal dataset
        if label and random.random() < syntheticTestProportion:
            syntheticBenchmarkDataset.append(newDatapoint)
        else:
            kgDataset[user].append(newDatapoint)

    if len(goodChoices) > 0:
        for i in range(random.randint(0, len(goodChoices) - 1)):
            generateSyntheticCompletion(goodChoices[i], True)
    if len(badChoices) > 0:
        for i in range(random.randint(0, len(badChoices) - 1)):
            generateSyntheticCompletion(badChoices[i], False)
    if len(repChoices) > 0:
        for i in range(random.randint(0, len(repChoices) - 1)):
            generateSyntheticCompletion(repChoices[i], False, False)


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

with open("nonFederatedMovieKnowledgeGraphDatasetWithSyntheticData.jsonl", "w") as file:
    for dataPoint in nonFederatedSyntheticDataset:
        file.write(json.dumps(dataPoint) + '\n')


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


# TODO: FIX REPEATED ENTRIES IN KNOWLEDGE GRAPH
