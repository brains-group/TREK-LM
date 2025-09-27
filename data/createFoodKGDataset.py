from enum import Enum
import kagglehub
import pickle
import os
from rdflib import Graph, URIRef, Literal
import argparse
import pandas as pd
import json
import random
from copy import deepcopy
from tqdm import tqdm

# ----------------- creating the knowledge graphs -----------------

URI_BASE = "http://pkg.org/"
USER_URI_BASE = URI_BASE + "user/"
RECIPE_URI_BASE = URI_BASE + "recipe/"

minRatings = 20
maxRatings = 1000


class Relation(Enum):
    TYPE = URIRef("@type")
    # for recipes
    NAME = URIRef("name")
    MINUTES = URIRef("minutesToCook")
    AUTHOR_ID = URIRef("authorID")
    SUBMISSION_DATE = URIRef("uploadDate")
    HAS_TAG = URIRef("hasTag")
    CALORIES = URIRef("numCalories")
    TOTAL_FAT = URIRef("totalFatPercentDailyValue")
    SUGAR = URIRef("sugarPercentDailyValue")
    SODIUM = URIRef("sodiumPercentDailyValue")
    PROTEIN = URIRef("proteinPercentDailyValue")
    SAT_FAT = URIRef("saturatedFatPercentDailyValue")
    CARBS = URIRef("carbohydratesPercentDailyValue")
    STEPS = URIRef("numRecipeSteps")
    HAS_INGREDIENT = URIRef("hasIngredient")
    # ratings
    RATINGS = [
        URIRef("0StarRating"),
        URIRef("1StarRating"),
        URIRef("2StarRating"),
        URIRef("3StarRating"),
        URIRef("4StarRating"),
        URIRef("5StarRating"),
    ]


class Type(Enum):
    RECIPE = URIRef("recipe")
    USER = URIRef("user")


# Download latest version
path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")

print("Path to dataset files:", path)


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
        id = URIRef(RECIPE_URI_BASE + str(recipeRow["name"]).replace(" ", "_"))

        kg.add(
            (
                id,
                Relation.TYPE.value,
                Type.RECIPE.value,
            )
        )
        kg.add(
            (
                id,
                Relation.NAME.value,
                Literal(recipeRow["name"]),
            )
        )
        kg.add(
            (
                id,
                Relation.MINUTES.value,
                Literal(recipeRow["minutes"]),
            )
        )
        kg.add(
            (
                id,
                Relation.AUTHOR_ID.value,
                Literal(recipeRow["contributor_id"]),
            )
        )
        kg.add(
            (
                id,
                Relation.SUBMISSION_DATE.value,
                Literal(recipeRow["submitted"]),
            )
        )
        tags = json.loads(recipeRow["tags"].replace("'", '"'))
        for tag in tags:
            kg.add(
                (
                    id,
                    Relation.HAS_TAG.value,
                    Literal(tag),
                )
            )
        nutritions = json.loads(recipeRow["nutrition"])
        for index in range(len(nutritions)):
            nutritionFact = nutritions[index]
            relation = [
                Relation.CALORIES,
                Relation.TOTAL_FAT,
                Relation.SUGAR,
                Relation.SODIUM,
                Relation.PROTEIN,
                Relation.SAT_FAT,
                Relation.CARBS,
            ][index].value
            kg.add(
                (
                    id,
                    relation,
                    Literal(nutritionFact),
                )
            )
        kg.add(
            (
                id,
                Relation.STEPS.value,
                Literal(recipeRow["n_steps"]),
            )
        )

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
        id = next(kg[: Relation.TYPE.value : Type.RECIPE.value])

        ingredients = json.loads(ppRecipeRow["ingredient_ids"])
        for ingredient in ingredients:
            kg.add(
                (
                    id,
                    Relation.HAS_INGREDIENT.value,
                    Literal(ingrList[ingredient]),
                )
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
        id = URIRef(USER_URI_BASE + str(userRow.name))

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

        kg.add((id, Relation.TYPE.value, Type.USER.value))

        for recipeID, rating in zip(recipes, ratings):
            kg.add(
                (
                    id,
                    Relation.RATINGS.value[int(rating)],
                    next(
                        recipeKGs[recipeID][: Relation.TYPE.value : Type.RECIPE.value]
                    ),
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


# ----------------- creating the KTO datapoints -----------------

USER_STRING = "user"
ASSISTANT_STRING = "assistant"
SYSTEM_STRING = "system"

CONTENT_STRING = "content"
ROLE_STRING = "role"

PROMPT_STRING = "prompt"
COMPLETION_STRING = "completion"
LABEL_STRING = "label"

KG_PREFACE_STRING = "You perform Knowledge Graph Completion. You will recommend a new triple to add to the user's knowledge graph with a tail entity that isn't already in their knowledge graph. The user's entity is represented by {}. Use this knowledge graph when responding to their queries: {}"
REQUEST_STRING = "Recommend a recipe with trait of {} -> {}."
COMPLETION_FORMAT_STRING = (
    "Based on your Knowledge Graph, I recommend the following:\n{}"
)

kgDataset = {}
recipeIDToIndex = {
    next(recipeKG[: Relation.TYPE.value : Type.RECIPE.value]): index
    for index, recipeKG in enumerate(recipeKGs)
}
sumPositiveDataPoints = 0
sumNegativeDataPoints = 0
for user, userKG in tqdm(enumerate(userKGs)):
    if userKG is None:
        continue

    def generateSyntheticCompletion(choiceOptions, shouldDelete=True):
        choiceIndex = random.randint(0, len(choiceOptions) - 1)
        startIndex = choiceOptions[choiceIndex]
        specificKG = deepcopy(userKG)
        triples = [triple for triple in userKG if triple[1] != Relation.TYPE.value]
        startRecipeKG = recipeKGs[recipeIDToIndex[triples[startIndex][2]]]
        trait = random.choice([Relation.HAS_TAG.value, Relation.HAS_INGREDIENT.value])
        traitValue = next(startRecipeKG[:trait:])[1]
        if shouldDelete:
            choiceOptions = choiceOptions[(choiceIndex + 1) :]
            for index in range(len(triples) - 1, startIndex - 1, -1):
                specificKG.remove(triples[index])

        choiceIndexes = [startIndex]
        if len(choiceOptions) > 0:
            choiceIndexes += random.sample(
                choiceOptions, random.randint(0, min(10, len(choiceOptions)))
            )

        goodRecommendations = ""
        badRecommendations = ""
        for choiceIndex in choiceIndexes:
            recommendation = triples[choiceIndex][2]
            if (
                shouldDelete
                and int(triples[choiceIndex][1][0]) > 2.5
                and next(recipeKGs[recipeIDToIndex[recommendation]][:trait:])[1]
                == traitValue
            ):
                goodRecommendations += "- " + str(recommendation) + "\n"
            else:
                badRecommendations += str(recommendation) + " "

        def makeDataPoint(recommendations, label):
            completion = [
                {
                    CONTENT_STRING: (
                        COMPLETION_FORMAT_STRING if label else "{}"
                    ).format(recommendations),
                    ROLE_STRING: ASSISTANT_STRING,
                }
            ]

            newDatapoint = {
                PROMPT_STRING: [
                    {
                        CONTENT_STRING: KG_PREFACE_STRING.format(
                            user, specificKG.serialize(format="json-ld")
                        ),
                        ROLE_STRING: SYSTEM_STRING,
                    },
                    {
                        CONTENT_STRING: REQUEST_STRING.format(trait, traitValue),
                        ROLE_STRING: USER_STRING,
                    },
                ],
                COMPLETION_STRING: completion,
                LABEL_STRING: label,
            }

            return newDatapoint

        if goodRecommendations != "":
            kgDataset[user].append(makeDataPoint(goodRecommendations, True))
            global sumPositiveDataPoints
            sumPositiveDataPoints += 1
        if badRecommendations != "":
            kgDataset[user].append(makeDataPoint(badRecommendations, False))
            global sumNegativeDataPoints
            sumNegativeDataPoints += 1

    choices = list(range(len(userKG) - 1))
    if len(choices) > 0:
        kgDataset[user] = []
        for i in range(random.randint(0, max(int(len(choices) / 10), 10))):
            generateSyntheticCompletion(choices)
        # avoid redundant recommendations
        for i in range(random.randint(0, max(int(len(choices) / 20), 10))):
            generateSyntheticCompletion(choices, False)

        numPos = sum([entry["label"] for entry in kgDataset[user]])
        numDataPoints = len(kgDataset[user])
        if numDataPoints < 10 or numPos == 0 or numPos == numDataPoints:
            del kgDataset[user]

print("--------- Base KG Dataset ---------")
print("Number of users: " + str(len(kgDataset.keys())))
print("Number of data points: " + str(sumPositiveDataPoints + sumNegativeDataPoints))
print("Number of positive data points: " + str(sumPositiveDataPoints))
print("Number of negative data points: " + str(sumNegativeDataPoints))
print(
    "Positive to Negative Ratio: " + str(sumPositiveDataPoints / sumNegativeDataPoints)
)

with open("recipeKGDataset.json", "w") as file:
    json.dump(kgDataset, file, indent=4)

nonFederatedDataset = []
for user in kgDataset.values():
    nonFederatedDataset.extend(user)

with open("nonFederatedRecipeKGDataset.json", "w") as file:
    json.dump(nonFederatedDataset, file, indent=4)
