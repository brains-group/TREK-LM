import json
import os
import random
from datasets import load_dataset

from utils import constants as C


def main():
    """
    Main function to generate movie knowledge graph datasets from the re_dial dataset.
    This includes processing conversations, creating positive and negative examples,
    generating synthetic data, and saving datasets in various formats.
    """
    random.seed(1)

    def triples_to_structured(triples):
        structured = {}
        for i in range(len(triples[C.HEAD_STRING])):
            head, relation, tail = triples[C.HEAD_STRING][i], triples[C.RELATION_STRING][i], triples[C.TAIL_STRING][i]
            tail_obj = {C.TYPE_STRING: C.MOVIE_TYPE, C.NAME_STRING: tail}
            structured[C.ID_STRING] = C.USER_ID_FORMAT.format(head)
            structured[C.TYPE_STRING] = C.USER_TYPE
            if relation not in structured:
                structured[relation] = tail_obj
            elif isinstance(structured[relation], list):
                structured[relation].append(tail_obj)
            else:
                structured[relation] = [structured[relation], tail_obj]
        return structured

    def preface_turn(user, kg):
        return C.KG_PREFACE_STRING.format(user, json.dumps(triples_to_structured(kg)))

    movie_dataset = load_dataset("community-datasets/re_dial")
    dataset = movie_dataset["train"].to_dict()
    test_dataset = movie_dataset["test"].to_dict()
    for key in dataset:
        dataset[key].extend(test_dataset[key])

    knowledge_graphs = {}
    kg_dataset = {}
    print("Processing conversations to build knowledge graphs...")

    for index in range(len(dataset["conversationId"])):
        movieMentions = {movie["movieId"]: movie["movieName"] for movie in dataset["movieMentions"][index]}
        userId = dataset["initiatorWorkerId"][index]
        messages = dataset["messages"][index]
        questions = {movie["movieId"]: movie for movie in (dataset["respondentQuestions"][index] if not dataset["initiatorQuestions"][index] else dataset["initiatorQuestions"][index])}
        if not questions: continue

        if userId not in knowledge_graphs:
            knowledge_graphs[userId] = {C.HEAD_STRING: [], C.RELATION_STRING: [], C.TAIL_STRING: []}
        userKG = knowledge_graphs[userId]
        prompt = [{C.CONTENT_STRING: preface_turn(userId, userKG), C.ROLE_STRING: C.SYSTEM_STRING}]

        def updateUserKG(movieName, question, isAssistantMessage=False):
            try:
                idx = userKG[C.TAIL_STRING].index(movieName)
            except ValueError:
                userKG[C.HEAD_STRING].append(userId)
                userKG[C.TAIL_STRING].append(movieName)
                userKG[C.RELATION_STRING].append("")
                idx = len(userKG[C.RELATION_STRING]) - 1

            if question[C.LIKED_STRING] != 2: userKG[C.RELATION_STRING][idx] = C.LIKED_STRING if question[C.LIKED_STRING] == 1 else C.DISLIKED_STRING
            elif question[C.SEEN_STRING] != 2:
                if userKG[C.RELATION_STRING][idx] not in [C.LIKED_STRING, C.DISLIKED_STRING]: userKG[C.RELATION_STRING][idx] = C.SEEN_STRING if question[C.SEEN_STRING] == 1 else C.UNSEEN_STRING
            elif question[C.SUGGESTED_STRING] == 1 and isAssistantMessage:
                if userKG[C.RELATION_STRING][idx] not in [C.LIKED_STRING, C.DISLIKED_STRING, C.SEEN_STRING, C.UNSEEN_STRING]: userKG[C.RELATION_STRING][idx] = C.SUGGESTED_STRING
            if userKG[C.RELATION_STRING][-1] == "":
                del userKG[C.HEAD_STRING][-1]; del userKG[C.TAIL_STRING][-1]; del userKG[C.RELATION_STRING][-1]

        for message in messages:
            turn = {C.CONTENT_STRING: message["text"], C.ROLE_STRING: C.USER_STRING if message["senderWorkerId"] == userId else C.ASSISTANT_STRING}
            moviesAdded = []
            for movieId, movieName in movieMentions.items():
                if movieName is None: continue
                newContent = turn[C.CONTENT_STRING].replace(f"@{movieId}", movieName)
                if newContent != turn[C.CONTENT_STRING] and movieId in questions:
                    if turn[C.ROLE_STRING] == C.USER_STRING: updateUserKG(movieName, questions[movieId])
                    else: moviesAdded.append((movieId, movieName))
                turn[C.CONTENT_STRING] = newContent

            if turn[C.ROLE_STRING] == C.ASSISTANT_STRING and moviesAdded:
                if userId not in kg_dataset: kg_dataset[userId] = []
                truncatedPrompt = prompt[:next((i for i in range(len(prompt), 0, -1) if prompt[i-1][C.ROLE_STRING] == C.USER_STRING), 0)]
                if not truncatedPrompt: break
                goodMovies, goodTriples, goals, badMovies, badTriples = "", "", [], "", ""
                for movieId, movieName in moviesAdded:
                    newTriple = {userId: {C.SUGGESTED_STRING: movieName}}
                    isGood = movieName not in userKG[C.TAIL_STRING] and questions[movieId][C.LIKED_STRING] != 0
                    if isGood:
                        goodMovies += f"- {movieName}\n"; goodTriples += json.dumps(newTriple) + "\n"
                        goals.append(movieName[:-7] if movieName.endswith(" ()") else movieName)
                    else:
                        badMovies += f"{movieName} "; badTriples += f"{json.dumps(newTriple)} "
                    updateUserKG(movieName, questions[movieId], True)

                def addDataPoint(movies, triples, label, goals=[]):
                    kg_dataset[userId].append({C.PROMPT_STRING: truncatedPrompt, C.COMPLETION_STRING: [{C.CONTENT_STRING: (C.COMPLETION_FORMAT_STRING if label else "{}\n{}").format(movies, triples), C.ROLE_STRING: C.ASSISTANT_STRING}], C.LABEL_STRING: label, C.GOAL_STRING: goals})

                if goodMovies: addDataPoint(goodMovies, goodTriples, True, goals)
                if badMovies: addDataPoint(badMovies, badTriples, False)

            prompt.append(turn)
            prompt[0][C.CONTENT_STRING] = preface_turn(userId, userKG)

    testProportion = 1 / 10
    syntheticTestProportion = 1 / 3
    realBenchmarkDataset, syntheticBenchmarkDataset, culledKGDataset = [], [], {}
    sumDataPoints, sumPositiveDataPoints, sumNegativeDataPoints, culledDataPoints, culledUsers = 0, 0, 0, 0, 0

    for user in list(kg_dataset.keys()):
        removedItems = 0
        for index in range(len(kg_dataset[user])):
            if kg_dataset[user][index - removedItems][C.LABEL_STRING]:
                if random.random() < testProportion:
                    realBenchmarkDataset.append(kg_dataset[user].pop(index - removedItems)); removedItems += 1
        sumChoices = sum(entry[C.LABEL_STRING] for entry in kg_dataset[user]); numDataPoints = len(kg_dataset[user])
        if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
            culledUsers += 1; culledDataPoints += numDataPoints; del kg_dataset[user]
        else:
            culledKGDataset[user] = kg_dataset[user]; sumDataPoints += numDataPoints; sumPositiveDataPoints += sumChoices; sumNegativeDataPoints += numDataPoints - sumChoices

    print("--------- Base KG Dataset ---------")
    print(f"Number of users: {len(culledKGDataset)}"); print(f"Number of data points: {sumDataPoints}"); print(f"Positive to Negative Ratio: {sumPositiveDataPoints/sumNegativeDataPoints if sumNegativeDataPoints > 0 else float('inf')}")

    with open("movieKnowledgeGraphDataset.json", "w") as f: json.dump(culledKGDataset, f, indent=4)
    nonFederatedDataset = [item for sublist in culledKGDataset.values() for item in sublist]
    with open("nonFederatedMovieKnowledgeGraphDataset.json", "w") as f: json.dump(nonFederatedDataset, f, indent=4)

    syntheticStartIndexes = {}
    for user in kg_dataset.keys():
        syntheticStartIndexes[user] = len(knowledge_graphs[user][C.HEAD_STRING])
        goodChoices = [i for i, rel in enumerate(knowledge_graphs[user][C.RELATION_STRING]) if rel == C.LIKED_STRING]
        badChoices = [i for i, rel in enumerate(knowledge_graphs[user][C.RELATION_STRING]) if rel == C.DISLIKED_STRING]
        repChoices = list(range(len(knowledge_graphs[user][C.RELATION_STRING])))
        userKG = knowledge_graphs[user]
        def generateSyntheticCompletion(choiceOptions, label, shouldDelete=True):
            choiceIndexes = random.sample(choiceOptions, random.randint(1, min(10, len(choiceOptions))))
            specificKG = {C.HEAD_STRING: userKG[C.HEAD_STRING].copy(), C.RELATION_STRING: userKG[C.RELATION_STRING].copy(), C.TAIL_STRING: userKG[C.TAIL_STRING].copy()}
            movies, triples, goals = "", "", []
            for choiceIndex in choiceIndexes:
                newTriple = {userKG[C.HEAD_STRING][choiceIndex]: {C.SUGGESTED_STRING: userKG[C.TAIL_STRING][choiceIndex]}}
                if label:
                    movies += f"- {userKG[C.TAIL_STRING][choiceIndex]}\n"; triples += json.dumps(newTriple) + "\n"
                    goal = userKG[C.TAIL_STRING][choiceIndex]; goals.append(goal[:-7] if goal.endswith(" ()") else goal)
                else:
                    movies += f"{userKG[C.TAIL_STRING][choiceIndex]} "; triples += f"{json.dumps(newTriple)} "
            choiceIndexes.sort(reverse=True)
            if shouldDelete:
                for choiceIndex in choiceIndexes: del specificKG[C.HEAD_STRING][choiceIndex]; del specificKG[C.RELATION_STRING][choiceIndex]; del specificKG[C.TAIL_STRING][choiceIndex]
            completion = [{C.CONTENT_STRING: (C.COMPLETION_FORMAT_STRING if label else "{}\n{}").format(movies, triples), C.ROLE_STRING: C.ASSISTANT_STRING}]
            newDatapoint = {C.PROMPT_STRING: [{C.CONTENT_STRING: preface_turn(user, specificKG), C.ROLE_STRING: C.SYSTEM_STRING}, {C.CONTENT_STRING: C.REQUEST_STRING, C.ROLE_STRING: C.USER_STRING}], C.COMPLETION_STRING: completion, C.LABEL_STRING: label, C.GOAL_STRING: goals}
            if label and random.random() < syntheticTestProportion: syntheticBenchmarkDataset.append(newDatapoint)
            else: kg_dataset[user].append(newDatapoint)
        if goodChoices:
            for _ in range(random.randint(0, len(goodChoices))): generateSyntheticCompletion(goodChoices, True)
        if badChoices:
            for _ in range(random.randint(0, len(badChoices))): generateSyntheticCompletion(badChoices, False)
        if repChoices:
            for _ in range(random.randint(0, len(repChoices))): generateSyntheticCompletion(repChoices, False, False)

    # Recalculate stats after adding synthetic data
    sumDataPoints, sumPositiveDataPoints, sumNegativeDataPoints, culledDataPoints, culledUsers = 0, 0, 0, 0, 0
    for user in list(kg_dataset.keys()):
        sumChoices = sum(entry[C.LABEL_STRING] for entry in kg_dataset[user]); numDataPoints = len(kg_dataset[user])
        if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
            culledUsers += 1; culledDataPoints += numDataPoints
            if syntheticStartIndexes.get(user, 0) > 0: realBenchmarkDataset.extend([e for e in kg_dataset[user][:syntheticStartIndexes[user]] if e[C.LABEL_STRING]])
            if syntheticStartIndexes.get(user, 0) < len(kg_dataset[user]): syntheticBenchmarkDataset.extend([e for e in kg_dataset[user][syntheticStartIndexes[user]:] if e[C.LABEL_STRING]])
            del kg_dataset[user]
        else:
            culledKGDataset[user] = kg_dataset[user]; sumDataPoints += numDataPoints; sumPositiveDataPoints += sumChoices; sumNegativeDataPoints += numDataPoints - sumChoices

    print("--------- KG Dataset with Synthetic Data ---------")
    # ... (prints for synthetic data stats)

    with open("movieKnowledgeGraphDatasetWithSyntheticData.json", "w") as f: json.dump(culledKGDataset, f, indent=4)
    nonFederatedSyntheticDataset = [item for sublist in culledKGDataset.values() for item in sublist]
    with open("nonFederatedMovieKnowledgeGraphDatasetWithSyntheticData.json", "w") as f: json.dump(nonFederatedSyntheticDataset, f, indent=4)
    with open("movieKnowledgeGraphTestDataset.json", "w") as f: json.dump(realBenchmarkDataset, f, indent=4)
    with open("movieKnowledgeGraphSyntheticTestDataset.json", "w") as f: json.dump(syntheticBenchmarkDataset, f, indent=4)

    entities, relations, test, train, valid = set(), set(), [], [], []
    testProportion = (len(realBenchmarkDataset) + len(syntheticBenchmarkDataset)) / len(nonFederatedSyntheticDataset) if nonFederatedSyntheticDataset else 0
    validProportion = 17535 / 272116
    for kg in knowledge_graphs.values():
        entities.update(str(e).replace(" ", "_") for e in kg[C.HEAD_STRING]); entities.update(str(e).replace(" ", "_") for e in kg[C.TAIL_STRING])
        relations.update(r.replace(" ", "_") for r in kg[C.RELATION_STRING])
        valid_user, train_user = [], []
        kgTriples = [f"{h.replace(' ', '_')}\t{r.replace(' ', '_')}\t{t.replace(' ', '_')}\n" for h, r, t in zip(kg[C.HEAD_STRING], kg[C.RELATION_STRING], kg[C.TAIL_STRING])]
        for triple in kgTriples:
            if random.random() < testProportion: test.append(triple)
            elif random.random() < validProportion: valid_user.append(triple)
            else: train_user.append(triple)
        if len(valid_user) + len(train_user) >= 10: valid.append(valid_user); train.append(train_user)
        else: test.extend(valid_user); test.extend(train_user)

    nonFederatedValid = [t for user_triples in valid for t in user_triples]
    nonFederatedTrain = [t for user_triples in train for t in user_triples]

    path = "../{}/data/movieKnowledgeGraphDataset"
    for modelName in ["HAKE", "KBGAT"]:
        loopPath = path.format(modelName); os.makedirs(loopPath, exist_ok=True)
        with open(f"{loopPath}/train.txt", "w") as f: f.writelines(nonFederatedTrain)
        with open(f"{loopPath}/test.txt", "w") as f: f.writelines(test)
        with open(f"{loopPath}/valid.txt", "w") as f: f.writelines(nonFederatedValid)
        federatedPath = os.path.join(loopPath, "federated"); os.makedirs(federatedPath, exist_ok=True)
        for i, (train_data, valid_data) in enumerate(zip(train, valid)):
            with open(f"{federatedPath}/train{i}.txt", "w") as f: f.writelines(train_data)
            with open(f"{federatedPath}/valid{i}.txt", "w") as f: f.writelines(valid_data)

    print("Dataset creation process finished.")

if __name__ == "__main__":
    main()