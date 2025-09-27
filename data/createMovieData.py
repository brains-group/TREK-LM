import sys
import os
from itertools import chain
import json
import random
from datasets import load_dataset
from rdflib import Graph, Literal
from rdflib.namespace import RDF, RDFS
from tqdm import tqdm  # Import tqdm

# Add the parent directory to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import constants as C
from utils.kg_creation import (
    add_data_point,
    generate_synthetic_completion,
    preface_turn,
    update_user_kg,
)
from utils.uri_helpers import get_movie_uri, get_relation_uri, get_user_uri


def main():
    """
    Main function to generate movie knowledge graph datasets from the re_dial dataset.
    This includes processing conversations, creating positive and negative examples,
    generating synthetic data, and saving datasets in various formats.
    """
    print("Starting dataset creation process...")
    random.seed(1)

    print("Loading movie dataset from \"community-datasets/re_dial\"...")
    movie_dataset = load_dataset("community-datasets/re_dial")
    dataset = movie_dataset["train"].to_dict()
    test_dataset = movie_dataset["test"].to_dict()
    for key in dataset:
        dataset[key].extend(test_dataset[key])
    print(f"Loaded {len(dataset['conversationId'])} conversations.")

    knowledge_graphs = {}
    kg_dataset = {}
    print("Processing conversations to build knowledge graphs...")

    # Wrap the conversation processing loop with tqdm
    for index in tqdm(range(len(dataset["conversationId"])), desc="Processing conversations"):
        movieMentions = {
            movie["movieId"]: movie["movieName"]
            for movie in dataset["movieMentions"][index]
        }
        userId = dataset["initiatorWorkerId"][index]
        user_uri = get_user_uri(userId)

        messages = dataset["messages"][index]
        questions = {
            movie["movieId"]: movie
            for movie in (
                dataset["respondentQuestions"][index]
                if not dataset["initiatorQuestions"][index]
                else dataset["initiatorQuestions"][index]
            )
        }
        if not questions:
            continue

        if userId not in knowledge_graphs:
            g = Graph()
            g.add((user_uri, RDF.type, C.USER_TYPE))
            knowledge_graphs[userId] = g
        g = knowledge_graphs[userId]

        prompt = [
            {
                C.CONTENT_STRING: preface_turn(str(user_uri), g),
                C.ROLE_STRING: C.SYSTEM_STRING,
            }
        ]

        for message in messages:
            turn = {
                C.CONTENT_STRING: message["text"],
                C.ROLE_STRING: (
                    C.USER_STRING
                    if message["senderWorkerId"] == userId
                    else C.ASSISTANT_STRING
                ),
            }
            moviesAdded = []
            for movieId, movieName in movieMentions.items():
                if movieName is None:
                    continue
                newContent = turn[C.CONTENT_STRING].replace(f"@{movieId}", movieName)
                if newContent != turn[C.CONTENT_STRING] and movieId in questions:
                    if turn[C.ROLE_STRING] == C.USER_STRING:
                        update_user_kg(
                            g, user_uri, movieName, questions[movieId], False
                        )
                    else:
                        moviesAdded.append((movieId, movieName))
                turn[C.CONTENT_STRING] = newContent

            if turn[C.ROLE_STRING] == C.ASSISTANT_STRING and moviesAdded:
                if userId not in kg_dataset:
                    kg_dataset[userId] = []
                truncatedPrompt = prompt[
                    : next(
                        (
                            i
                            for i in range(len(prompt), 0, -1)
                            if prompt[i - 1][C.ROLE_STRING] == C.USER_STRING
                        ),
                        0,
                    )
                ]
                if not truncatedPrompt:
                    break
                goodMovies, goals, badMovies = "", [], ""
                for movieId, movieName in moviesAdded:
                    movie_uri = get_movie_uri(movieName)
                    try:
                        next(g[user_uri::movie_uri])
                        dupe = True
                    except StopIteration:
                        dupe = False
                    isGood = not dupe and questions[movieId][C.LIKED_STRING] != 0

                    if isGood:
                        goodMovies += f"- {movieName}\n"
                        goals.append(
                            movieName[: movieName.rfind(" ")]
                            if movieName.endswith(")")
                            else movieName
                        )
                    else:
                        badMovies += f"{movieName} "
                    update_user_kg(g, user_uri, movieName, questions[movieId], True)

                if goodMovies:
                    add_data_point(
                        kg_dataset, userId, truncatedPrompt, goodMovies, True, goals
                    )
                if badMovies:
                    add_data_point(
                        kg_dataset, userId, truncatedPrompt, badMovies, False
                    )

            prompt.append(turn)
            prompt[0][C.CONTENT_STRING] = preface_turn(str(user_uri), g)

    testProportion = 1 / 10
    syntheticTestProportion = 1 / 3
    realBenchmarkDataset, syntheticBenchmarkDataset, culledKGDataset = [], [], {}
    (
        sumDataPoints,
        sumPositiveDataPoints,
        sumNegativeDataPoints,
        culledDataPoints,
        culledUsers,
    ) = (0, 0, 0, 0, 0)

    print("Culling and splitting dataset...")
    for user in tqdm(list(kg_dataset.keys()), desc="Culling users"):
        removedItems = 0
        for index in range(len(kg_dataset[user])):
            if kg_dataset[user][index - removedItems][C.LABEL_STRING]:
                if random.random() < testProportion:
                    realBenchmarkDataset.append(
                        kg_dataset[user].pop(index - removedItems)
                    )
                    removedItems += 1
        sumChoices = sum(entry[C.LABEL_STRING] for entry in kg_dataset[user])
        numDataPoints = len(kg_dataset[user])
        if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
            culledUsers += 1
            culledDataPoints += numDataPoints
        else:
            culledKGDataset[user] = kg_dataset[user]
            sumDataPoints += numDataPoints
            sumPositiveDataPoints += sumChoices
            sumNegativeDataPoints += numDataPoints - sumChoices

    # Prepare for outputting stats to a file
    stats_filename = "dataset_stats.txt"
    print(f"Outputting dataset statistics to {stats_filename}")

    with open(stats_filename, "w") as stats_file:
        stats_file.write("--------- Base KG Dataset ---------\n")
        stats_file.write(f"Number of users: {len(culledKGDataset)}\n")
        stats_file.write(f"Number of data points: {sumDataPoints}\n")
        stats_file.write(f"Number of positive data points: {sumPositiveDataPoints}\n")
        stats_file.write(f"Number of negative data points: {sumNegativeDataPoints}\n")
        ratio = sumPositiveDataPoints/sumNegativeDataPoints if sumNegativeDataPoints > 0 else float('inf')
        stats_file.write(f"Positive to Negative Ratio: {ratio}\n")
        stats_file.write(f"Number of culled users: {culledUsers}\n")
        stats_file.write(f"Number of culled datapoints: {culledDataPoints}\n")

        print("--------- Base KG Dataset ---------")
        print(f"Number of users: {len(culledKGDataset)}")
        print(f"Number of data points: {sumDataPoints}")
        print(f"Number of positive data points: {sumPositiveDataPoints}")
        print(f"Number of negative data points: {sumNegativeDataPoints}")
        print(
            f"Positive to Negative Ratio: {sumPositiveDataPoints/sumNegativeDataPoints if sumNegativeDataPoints > 0 else float('inf')}"
        )
        print(f"Number of culled users: {culledUsers}")
        print(f"Number of culled datapoints: {culledDataPoints}")

        print("Saving base KG dataset to JSON files...")
        with open("movieKnowledgeGraphDataset.json", "w") as f:
            json.dump(culledKGDataset, f, indent=4)
        nonFederatedDataset = [
            item for sublist in culledKGDataset.values() for item in sublist
        ]
        with open("nonFederatedMovieKnowledgeGraphDataset.json", "w") as f:
            json.dump(nonFederatedDataset, f, indent=4)
        print("Base KG dataset saved.")

        print("Generating synthetic data...")
        syntheticStartIndexes = {}
        for user in tqdm(list(kg_dataset.keys()), desc="Generating synthetic data for users"):
            user_uri = get_user_uri(user)
            g = knowledge_graphs[user]
            syntheticStartIndexes[user] = (
                len(list(g.triples((user_uri, None, None)))) - 1
            )  # Exclude the user type triple

            # Get all movies the user has interacted with
            all_movies = list(g.subjects(predicate=RDF.type, object=C.MOVIE_TYPE))

            liked_movies = [
                o
                for s, p, o in g.triples((user_uri, get_relation_uri(C.LIKED_STRING), None))
            ]
            disliked_movies = [
                o
                for s, p, o in g.triples(
                    (user_uri, get_relation_uri(C.DISLIKED_STRING), None)
                )
            ]

            def process_synthetic_generation(movies, label, should_remove_relations=True):
                if not movies:
                    return
                for _ in range(random.randint(0, len(movies))):
                    new_datapoint = generate_synthetic_completion(
                        user_uri, g, movies, label, should_remove_relations
                    )
                    if new_datapoint:
                        if label and random.random() < syntheticTestProportion:
                            syntheticBenchmarkDataset.append(new_datapoint)
                        else:
                            kg_dataset[user].append(new_datapoint)

            process_synthetic_generation(liked_movies, True)
            process_synthetic_generation(disliked_movies, False)
            process_synthetic_generation(all_movies, False, should_remove_relations=False)

        print("Recalculating stats after adding synthetic data...")
        # Recalculate stats after adding synthetic data
        (
            sumDataPoints,
            sumPositiveDataPoints,
            sumNegativeDataPoints,
            culledDataPoints,
            culledUsers,
        ) = (0, 0, 0, 0, 0)
        for user in tqdm(list(kg_dataset.keys()), desc="Recalculating stats"):
            sumChoices = sum(entry[C.LABEL_STRING] for entry in kg_dataset[user])
            numDataPoints = len(kg_dataset[user])
            if numDataPoints < 10 or sumChoices == 0 or sumChoices == numDataPoints:
                culledUsers += 1
                culledDataPoints += numDataPoints
                if syntheticStartIndexes.get(user, 0) > 0:
                    realBenchmarkDataset.extend(
                        [
                            e
                            for e in kg_dataset[user][: syntheticStartIndexes[user]]
                            if e[C.LABEL_STRING]
                        ]
                    )
                if syntheticStartIndexes.get(user, 0) < len(kg_dataset[user]):
                    syntheticBenchmarkDataset.extend(
                        [
                            e
                            for e in kg_dataset[user][syntheticStartIndexes[user] :]
                            if e[C.LABEL_STRING]
                        ]
                    )
                del kg_dataset[user]
            else:
                culledKGDataset[user] = kg_dataset[user]
                sumDataPoints += numDataPoints
                sumPositiveDataPoints += sumChoices
                sumNegativeDataPoints += numDataPoints - sumChoices

        stats_file.write("\n--------- KG Dataset with Synthetic Data ---------\n")
        stats_file.write("Number of culled datapoints: " + str(culledDataPoints) + "\n")
        stats_file.write("Number of data points: " + str(sumDataPoints) + "\n")
        stats_file.write("Number of positive data points: " + str(sumPositiveDataPoints) + "\n")
        stats_file.write("Number of negative data points: " + str(sumNegativeDataPoints) + "\n")
        ratio = sumPositiveDataPoints / sumNegativeDataPoints if sumNegativeDataPoints > 0 else float('inf')
        stats_file.write("Positive to Negative Ratio: " + str(ratio) + "\n")
        stats_file.write("Number of culled users: " + str(culledUsers) + "\n")
        stats_file.write("Number of culled datapoints: " + str(culledDataPoints) + "\n")

        print("--------- KG Dataset with Synthetic Data ---------")
        print("Number of culled datapoints: " + str(culledDataPoints))
        print("Number of data points: " + str(sumDataPoints))
        print("Number of positive data points: " + str(sumPositiveDataPoints))
        print("Number of negative data points: " + str(sumNegativeDataPoints))
        print(
            "Positive to Negative Ratio: "
            + str(sumPositiveDataPoints / sumNegativeDataPoints)
        )
        print("Number of culled users: " + str(culledUsers))
        print("Number of culled datapoints: " + str(culledDataPoints))

        print("Saving KG dataset with synthetic data to JSON files...")
        with open("movieKnowledgeGraphDatasetWithSyntheticData.json", "w") as f:
            json.dump(culledKGDataset, f, indent=4)
        nonFederatedSyntheticDataset = [
            item for sublist in culledKGDataset.values() for item in sublist
        ]
        with open("nonFederatedMovieKnowledgeGraphDatasetWithSyntheticData.json", "w") as f:
            json.dump(nonFederatedSyntheticDataset, f, indent=4)
        print("KG dataset with synthetic data saved.")

        stats_file.write("\n--------- KG Test Dataset ---------\n")
        stats_file.write("Number of data points: " + str(len(realBenchmarkDataset)) + "\n")

        print("--------- KG Test Dataset ---------")
        print("Number of data points: " + str(len(realBenchmarkDataset)))
        print("Saving KG test dataset to JSON file...")
        with open("movieKnowledgeGraphTestDataset.json", "w") as f:
            json.dump(realBenchmarkDataset, f, indent=4)
        print("KG test dataset saved.")

        stats_file.write("\n--------- KG Synthetic Test Dataset ---------\n")
        stats_file.write("Number of data points: " + str(len(syntheticBenchmarkDataset)) + "\n")

        print("--------- KG Synthetic Test Dataset ---------")
        print("Number of data points: " + str(len(syntheticBenchmarkDataset)))
        print("Saving KG synthetic test dataset to JSON file...")
        with open("movieKnowledgeGraphSyntheticTestDataset.json", "w") as f:
            json.dump(syntheticBenchmarkDataset, f, indent=4)
        print("KG synthetic test dataset saved.")

    print("Dataset creation process finished. Statistics saved to " + stats_filename)


if __name__ == "__main__":
    main()
