import sys
import os
import kagglehub
import pickle
import pandas as pd
import json
import random
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF
import re
from tqdm import tqdm

# Add the parent directory to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import constants as C
from utils.models import get_tokenizer
from utils.data import (
    adapt_HAKE_and_KBGAT_and_FedKGRec_data,
    find_longest_tokenized_prompt,
)
from utils.food_kg_creation import (
    get_rating_uri,
    get_recipe_uri,
    add_data_point,
    generate_synthetic_completion,
)
from utils.uri_helpers import get_user_uri


minRatings = 20
maxRatings = 1000


def main():
    print("Starting dataset creation process...")
    random.seed(1)

    # Download latest version
    path = kagglehub.dataset_download(
        "shuyangli94/food-com-recipes-and-user-interactions"
    )
    print("Path to dataset files:", path)

    # Initialize Qwen3 tokenizer
    print("Loading Qwen3 tokenizer...")
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B", use_fast=False, padding_side="left")
    print("Qwen3 tokenizer loaded.")

    recipeKGsPath = "./recipeKGs.pkl"
    if not os.path.exists(recipeKGsPath):
        ingrList = (
            pd.read_csv(
                os.path.join(path + "/ingr_map.csv"),
                index_col="id",
                usecols=["replaced", "id"],
            )
            .drop_duplicates()
            .sort_index()["replaced"]
            .tolist()
        )
        print(f"Number of ingredients: {len(ingrList)}")

        def recipeRowToKG(recipeRow):
            kg = Graph()
            recipe_uri = get_recipe_uri(str(recipeRow["name"]))

            kg.add((recipe_uri, RDF.type, C.RECIPE_TYPE))
            kg.add((recipe_uri, C.NAME_STRING, Literal(recipeRow["name"])))
            kg.add((recipe_uri, C.MINUTES_STRING, Literal(recipeRow["minutes"])))
            kg.add(
                (recipe_uri, C.AUTHOR_ID_STRING, Literal(recipeRow["contributor_id"]))
            )
            kg.add(
                (recipe_uri, C.SUBMISSION_DATE_STRING, Literal(recipeRow["submitted"]))
            )
            tags = json.loads(recipeRow["tags"].replace("'", '"'))
            for tag in tags:
                kg.add((recipe_uri, C.HAS_TAG_STRING, Literal(tag)))
            nutritions = json.loads(recipeRow["nutrition"])
            nutrition_relations = [
                C.CALORIES_STRING,
                C.TOTAL_FAT_STRING,
                C.SUGAR_STRING,
                C.SODIUM_STRING,
                C.PROTEIN_STRING,
                C.SAT_FAT_STRING,
                C.CARBS_STRING,
            ]
            for index, nutritionFact in enumerate(nutritions):
                relation = nutrition_relations[index]
                kg.add((recipe_uri, relation, Literal(nutritionFact)))
            kg.add((recipe_uri, C.STEPS_STRING, Literal(recipeRow["n_steps"])))

            return kg

        recipeDF = pd.read_csv(
            os.path.join(path + "/RAW_recipes.csv"),
            index_col="id",
            usecols=[
                "name",
                "id",
                "minutes",
                "contributor_id",
                "submitted",
                "tags",
                "nutrition",
                "n_steps",
            ],
        )
        recipeKGsDF = recipeDF.apply(recipeRowToKG, axis=1)
        print(f"Number of recipes: {len(recipeKGsDF)}")

        def ppRecipeRowToKG(ppRecipeRow):
            kg = recipeKGsDF.loc[ppRecipeRow["id"]]
            recipe_uri = next(kg[: RDF.type : C.RECIPE_TYPE])

            ingredients = json.loads(ppRecipeRow["ingredient_ids"])
            for ingredient in ingredients:
                kg.add(
                    (recipe_uri, C.HAS_INGREDIENT_STRING, Literal(ingrList[ingredient]))
                )

            return kg

        ppRecipeDF = pd.read_csv(
            os.path.join(path + "/PP_recipes.csv"),
            index_col="i",
            usecols=[
                "id",
                "i",
                "ingredient_ids",
            ],
        ).sort_index()
        recipeKGs = ppRecipeDF.apply(ppRecipeRowToKG, axis=1).to_list()
        print(f"Number of pp recipes: {len(recipeKGs)}")
        with open(recipeKGsPath, "wb") as file:
            pickle.dump(recipeKGs, file)
    else:
        with open(recipeKGsPath, "rb") as file:
            recipeKGs = pickle.load(file)

    usersKGsPath = "./userKGs.pkl"
    if not os.path.exists(usersKGsPath):

        def userRowToKG(userRow):
            kg = Graph()
            user_uri = get_user_uri(str(userRow.name))

            recipes = json.loads(userRow["items"])
            ratings = json.loads(userRow["ratings"])
            positives = sum([rating > 2.5 for rating in ratings])
            if (
                len(ratings) < minRatings
                or len(ratings) > maxRatings
                or positives == 0
                or positives == len(ratings)
            ):
                return None

            kg.add((user_uri, RDF.type, C.USER_TYPE))

            for recipeID, rating in zip(recipes, ratings):
                kg.add(
                    (
                        user_uri,
                        get_rating_uri(int(rating)),
                        next(recipeKGs[recipeID][: RDF.type : C.RECIPE_TYPE]),
                    )
                )

            return kg

        userDF = pd.read_csv(
            os.path.join(path + "/PP_users.csv"),
            index_col="u",
            usecols=[
                "u",
                "items",
                "ratings",
            ],
        ).sort_index()
        userKGs = userDF.apply(userRowToKG, axis=1)
        print(f"Number of users: {len(userKGs)}")
        with open(usersKGsPath, "wb") as file:
            pickle.dump(userKGs, file)
    else:
        with open(usersKGsPath, "rb") as file:
            userKGs = pickle.load(file)

    kg_dataset = {}
    synthetic_benchmark_dataset = []
    synthetic_test_proportion = 1 / 10

    print("Generating synthetic data...")
    for user_index, user_kg in enumerate(
        tqdm(userKGs, desc="Generating synthetic data for users")
    ):
        if user_kg is None:
            continue

        user_uri = get_user_uri(str(user_index))

        # Get all recipes the user has interacted with (liked, disliked, or seen)
        all_user_recipes = [
            o for s, p, o in user_kg.triples((user_uri, None, None)) if o != C.USER_TYPE
        ]

        positive_recipes = []
        for i in range(3, 6):  # Ratings 3,4,5 for positive
            positive_recipes.extend(list(user_kg[user_uri : get_rating_uri(i) :]))

        negative_recipes = []
        for i in range(0, 3):  # Ratings 3,4,5 for positive; 0,1,2 for negative
            negative_recipes.extend(list(user_kg[user_uri : get_rating_uri(i) :]))

        # Generate positive synthetic data
        for _ in range(
            # random.randint(5, 15)
            random.randint(0, max(int(len(positive_recipes) / 10), 10))
        ):  # Generate a few positive examples per user
            new_datapoint = generate_synthetic_completion(
                user_uri,
                user_kg,
                positive_recipes,
                True,
                should_remove_relations=True,
            )
            if new_datapoint:
                if random.random() < synthetic_test_proportion:
                    synthetic_benchmark_dataset.append(new_datapoint)
                else:
                    add_data_point(kg_dataset, str(user_index), new_datapoint)

        # Generate negative synthetic data
        for _ in range(
            # random.randint(5, 15)
            random.randint(0, max(int(len(negative_recipes) / 10), 10))
        ):  # Generate a few positive examples per user
            new_datapoint = generate_synthetic_completion(
                user_uri,
                user_kg,
                negative_recipes,
                False,
                should_remove_relations=True,
            )
            if new_datapoint:
                add_data_point(kg_dataset, str(user_index), new_datapoint)

        # Generate negative redundant synthetic data
        for _ in range(
            # random.randint(5, 15)
            random.randint(0, max(int(len(negative_recipes) / 20), 10))
        ):  # Generate a few negative examples per user
            new_datapoint = generate_synthetic_completion(
                user_uri,
                user_kg,
                all_user_recipes,
                False,
                should_remove_relations=False,
            )
            if new_datapoint:
                add_data_point(kg_dataset, str(user_index), new_datapoint)

    # Recalculate stats and save datasets
    sum_data_points, sum_positive_data_points, sum_negative_data_points = 0, 0, 0
    culled_data_points, culled_users = 0, 0

    print("Culling and splitting dataset...")
    for user in tqdm(list(kg_dataset.keys()), desc="Culling users"):
        sum_choices = sum(entry[C.LABEL_STRING] for entry in kg_dataset[user])
        num_data_points = len(kg_dataset[user])
        if num_data_points < 20 or sum_choices == 0 or sum_choices == num_data_points:
            culled_users += 1
            culled_data_points += num_data_points
            synthetic_benchmark_dataset.extend(kg_dataset[user])
            del kg_dataset[user]
        else:
            sum_data_points += num_data_points
            sum_positive_data_points += sum_choices
            sum_negative_data_points += num_data_points - sum_choices

    stats_filename = "recipe_dataset_stats.txt"
    print(f"Outputting dataset statistics to {stats_filename}")

    with open(stats_filename, "w") as stats_file:
        stats_file.write("--------- Base KG Dataset ---------\n")
        stats_file.write(f"Number of users: {len(kg_dataset)}\n")
        stats_file.write(f"Number of data points: {sum_data_points}\n")
        stats_file.write(
            f"Number of positive data points: {sum_positive_data_points}\n"
        )
        stats_file.write(
            f"Number of negative data points: {sum_negative_data_points}\n"
        )
        ratio = (
            sum_positive_data_points / sum_negative_data_points
            if sum_negative_data_points > 0
            else float("inf")
        )
        stats_file.write(f"Positive to Negative Ratio: {ratio}\n")
        stats_file.write(f"Number of culled users: {culled_users}\n")
        stats_file.write(f"Number of culled datapoints: {culled_data_points}\n")

        print("--------- Base KG Dataset ---------")
        print(f"Number of users: {len(kg_dataset)}")
        print(f"Number of data points: {sum_data_points}")
        print(f"Number of positive data points: {sum_positive_data_points}")
        print(f"Number of negative data points: {sum_negative_data_points}")
        print(
            f"Positive to Negative Ratio: {sum_positive_data_points/sum_negative_data_points if sum_negative_data_points > 0 else float('inf')}"
        )
        print(f"Number of culled users: {culled_users}")
        print(f"Number of culled datapoints: {culled_data_points}")

        base_kg_longest_tokens = find_longest_tokenized_prompt(
            kg_dataset, tokenizer, "Base KG Dataset"
        )
        stats_file.write(
            f"Longest tokenized prompt (Base KG Dataset): {base_kg_longest_tokens}\n"
        )

        print("Saving base KG dataset to JSON files...")
        with open("recipeKGDataset.json", "w") as f:
            json.dump(kg_dataset, f, indent=4)
        non_federated_dataset = [
            item for sublist in kg_dataset.values() for item in sublist
        ]
        with open("nonFederatedRecipeKGDataset.json", "w") as f:
            json.dump(non_federated_dataset, f, indent=4)
        print("Base KG dataset saved.")

        stats_file.write("\n--------- KG Synthetic Test Dataset ---------\n")
        stats_file.write(
            "Number of data points: " + str(len(synthetic_benchmark_dataset)) + "\n"
        )

        print("--------- KG Synthetic Test Dataset ---------")
        print("Number of data points: " + str(len(synthetic_benchmark_dataset)))
        synthetic_benchmark_longest_tokens = find_longest_tokenized_prompt(
            synthetic_benchmark_dataset, tokenizer, "KG Synthetic Test Dataset"
        )
        stats_file.write(
            f"Longest tokenized prompt (KG Synthetic Test Dataset): {synthetic_benchmark_longest_tokens}\n"
        )

        print("Saving KG synthetic test dataset to JSON file...")
        with open("recipeKGTestDataset.json", "w") as f:
            json.dump(synthetic_benchmark_dataset, f, indent=4)
        print("KG synthetic test dataset saved.")

    print("Creating adapted datasets for KBGAT, HAKE, and FedKGRec...")

    testTotal = len(synthetic_benchmark_dataset)
    testProportion = testTotal / (len(non_federated_dataset) + testTotal)
    validProportion = 17535 / 272116
    adapt_HAKE_and_KBGAT_and_FedKGRec_data(
        userKGs,
        testProportion,
        validProportion,
        "recipeKnowledgeGraphDataset",
        lambda triple: re.sub(r"\trating(\d)\t", r"\t\1\t", triple),
        lambda triple: re.search(r"\trating\d\t", triple),
    )

    print("Dataset creation process finished. Statistics saved to " + stats_filename)


if __name__ == "__main__":
    main()
