from itertools import chain
import random
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

from utils import constants as C
from utils.uri_helpers import get_user_uri


def get_rating_uri(i):
    return URIRef(C.RELATION_RATING_PREFIX + str(i))


def get_recipe_uri(recipe_name):
    return URIRef(f"{C.RECIPE_PREFIX}{recipe_name.replace(' ', '_')}")


def add_data_point(dataset, user_id, datapoint):
    if user_id not in dataset:
        dataset[user_id] = []
    dataset[user_id].append(datapoint)


def format_recipe_completion(recipe_names, is_positive):
    if is_positive:
        completion_text = C.COMPLETION_FORMAT_STRING
        goals = recipe_names
    else:
        completion_text = ""
        goals = []

    completion = [
        {
            C.CONTENT_STRING: C.FOOD_COMPLETION_FORMAT_STRING.format(
                completion_text + "\n".join([f"- {name}" for name in recipe_names])
            ),
            C.ROLE_STRING: C.ASSISTANT_STRING,
        }
    ]
    return completion, goals


def preface_turn(user_uri, g):
    context = {
        "@vocab": C.EX,
        "rel": C.REL,
        "rdfs": C.RDFS_NS,
        "recipe": C.RECIPE_PREFIX,
        "user": C.USER_PREFIX,
        C.RELATION_RATING_PREFIX: f"rel:{C.RELATION_RATING_PREFIX}",
    }
    kg_json_str = g.serialize(format="json-ld", context=context)
    return C.FOOD_KG_PREFACE_STRING.format(user_uri[(len(C.USER_PREFIX)) :], kg_json_str)


def generate_synthetic_completion(
    user_uri,
    user_kg,
    all_user_recipes,
    # recipe_kgs,
    # recipe_id_to_index,
    label,
    should_remove_relations=True,
):
    if not all_user_recipes:
        return None

    # if not should_remove_relations:
    #     candidate_recipes = list(all_user_recipes)
    # else:
    #     candidate_recipes = []
    #     for i in (
    #         range(3, 6) if label else range(0, 3)
    #     ):  # Ratings 3,4,5 for positive; 0,1,2 for negative
    #         candidate_recipes.extend(list(user_kg[user_uri : get_rating_uri(i) :]))

    # if not candidate_recipes:
    #     return None
    candidate_recipes = list(all_user_recipes)

    # Select multiple random recipes
    num_recommendations = random.randint(
        1, min(10, len(candidate_recipes))
    )  # Recommend between 1 and 10 recipes
    chosen_recipe_uris = random.sample(candidate_recipes, num_recommendations)
    chosen_recipe_names = [
        str(uri).split("/")[-1].replace("_", " ") for uri in chosen_recipe_uris
    ]

    # Create a deep copy of the user's KG for this synthetic data point
    specific_kg = Graph()
    specific_kg += user_kg

    # Optionally remove relations to simulate a 'before recommendation' state
    if should_remove_relations:
        for recipe_uri in chosen_recipe_uris:
            for triple in chain(
                specific_kg.triples((None, None, recipe_uri)),
                specific_kg.triples((recipe_uri, None, None)),
            ):
                specific_kg.remove(triple)

    # Construct the prompt
    prompt = [
        {
            C.CONTENT_STRING: preface_turn(str(user_uri), specific_kg),
            C.ROLE_STRING: C.SYSTEM_STRING,
        },
        {
            C.CONTENT_STRING: C.FOOD_REQUEST_STRING,
            C.ROLE_STRING: C.USER_STRING,
        },
    ]

    # Format the completion based on the label
    completion, goals = format_recipe_completion(chosen_recipe_names, label)

    return {
        C.PROMPT_STRING: prompt,
        C.COMPLETION_STRING: completion,
        C.LABEL_STRING: label,
        C.GOAL_STRING: goals,
    }
