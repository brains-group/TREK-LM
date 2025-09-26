"""
This module contains utility functions for creating the knowledge graph dataset
from the raw movie recommendation dataset. It includes functions for processing
conversations, updating knowledge graphs, and generating synthetic data points.
"""

import random
from itertools import chain
from rdflib import Graph, Literal
from rdflib.namespace import RDF, RDFS

from utils import constants as C
from utils.uri_helpers import get_movie_uri, get_relation_uri


def preface_turn(user_uri, g):
    """
    Generates the preface string for a conversation turn, including the serialized knowledge graph.

    Args:
        user_uri (str): The URI of the user.
        g (Graph): The knowledge graph for the user.

    Returns:
        str: The preface string with the KG context.
    """
    context = {
        "@vocab": C.EX,
        "rel": C.REL,
        C.LIKED_STRING: f"rel:{C.LIKED_STRING}",
        C.DISLIKED_STRING: f"rel:{C.DISLIKED_STRING}",
        C.SEEN_STRING: f"rel:{C.SEEN_STRING}",
        C.UNSEEN_STRING: f"rel:{C.UNSEEN_STRING}",
        C.SUGGESTED_STRING: f"rel:{C.SUGGESTED_STRING}",
    }
    kg_json_str = g.serialize(format="json-ld", context=context)
    return C.KG_PREFACE_STRING.format(user_uri, kg_json_str)


def update_user_kg(g, user_uri, movie_name, question, is_assistant_message=False):
    """
    Updates the user's knowledge graph based on a movie mention and the user's response.

    Args:
        g (Graph): The user's knowledge graph.
        user_uri (URIRef): The URI of the user.
        movie_name (str): The name of the movie.
        question (dict): A dictionary containing the user's answers about the movie.
        is_assistant_message (bool): Whether the message is from the assistant.
    """
    movie_uri = get_movie_uri(movie_name)

    relations_priority = {
        get_relation_uri(C.LIKED_STRING): 3,
        get_relation_uri(C.DISLIKED_STRING): 3,
        get_relation_uri(C.SEEN_STRING): 2,
        get_relation_uri(C.UNSEEN_STRING): 2,
        get_relation_uri(C.SUGGESTED_STRING): 1,
    }

    new_relation = None
    if question[C.LIKED_STRING] != 2:
        new_relation = (
            get_relation_uri(C.LIKED_STRING)
            if question[C.LIKED_STRING] == 1
            else get_relation_uri(C.DISLIKED_STRING)
        )
    elif question[C.SEEN_STRING] != 2:
        new_relation = (
            get_relation_uri(C.SEEN_STRING)
            if question[C.SEEN_STRING] == 1
            else get_relation_uri(C.UNSEEN_STRING)
        )
    elif question[C.SUGGESTED_STRING] == 1 and is_assistant_message:
        new_relation = get_relation_uri(C.SUGGESTED_STRING)

    if not new_relation:
        return

    new_priority = relations_priority.get(new_relation, 0)
    try:
        existing_relation = next(g[user_uri::movie_uri])
    except StopIteration:
        existing_relation = None

    if existing_relation:
        existing_priority = relations_priority.get(existing_relation, 0)
        if new_priority >= existing_priority:
            g.remove((user_uri, existing_relation, movie_uri))
            g.add((user_uri, new_relation, movie_uri))
    else:
        g.add((movie_uri, RDF.type, C.MOVIE_TYPE))
        g.add((movie_uri, RDFS.label, Literal(movie_name)))
        g.add((user_uri, new_relation, movie_uri))


def add_data_point(kg_dataset, user_id, prompt, movies, label, goals=[]):
    """
    Adds a new data point to the knowledge graph dataset.

    Args:
        kg_dataset (dict): The dataset to add to.
        user_id (str): The ID of the user.
        prompt (list): The conversation prompt.
        movies (str): The movie completion string.
        label (bool): The label for the data point (True for good, False for bad).
        goals (list): A list of movie goals for positive examples.
    """
    if user_id not in kg_dataset:
        kg_dataset[user_id] = []

    completion_content = C.COMPLETION_FORMAT_STRING.format(movies) if label else movies
    kg_dataset[user_id].append(
        {
            C.PROMPT_STRING: prompt,
            C.COMPLETION_STRING: [
                {
                    C.CONTENT_STRING: completion_content,
                    C.ROLE_STRING: C.ASSISTANT_STRING,
                }
            ],
            C.LABEL_STRING: label,
            C.GOAL_STRING: goals,
        }
    )


def generate_synthetic_completion(
    user_uri,
    g,
    movies_to_include,
    label,
    should_remove_relations=True,
):
    """
    Generates a synthetic data point.

    Args:
        user_uri (URIRef): The URI of the user.
        g (Graph): The user's knowledge graph.
        movies_to_include (list): A list of movie URIs to sample from.
        label (bool): The label for the synthetic data point.
        should_remove_relations (bool): Whether to remove movie relations from the temp graph.

    Returns:
        dict or None: The generated synthetic data point, or None if no movies were provided.
    """
    if not movies_to_include:
        return None

    num_to_sample = random.randint(1, min(10, len(movies_to_include)))
    sampled_movies = random.sample(movies_to_include, num_to_sample)

    temp_g = Graph()
    temp_g += g

    movies_str, goals = "", []
    for movie_uri in sampled_movies:
        movie_name = str(temp_g.value(subject=movie_uri, predicate=RDFS.label))
        if label:
            movies_str += f"- {movie_name}\n"
            goals.append(
                movie_name[: movie_name.rfind(" ")]
                if movie_name.endswith(" ()")
                else movie_name
            )
        else:
            movies_str += f"{movie_name} "

        if should_remove_relations:
            for s, p, o in chain(
                temp_g.triples((movie_uri, None, None)),
                temp_g.triples((None, None, movie_uri)),
            ):
                temp_g.remove((s, p, o))

    completion_content = (
        C.COMPLETION_FORMAT_STRING.format(movies_str) if label else movies_str
    )
    return {
        C.PROMPT_STRING: [
            {
                C.CONTENT_STRING: preface_turn(str(user_uri), temp_g),
                C.ROLE_STRING: C.SYSTEM_STRING,
            },
            {C.CONTENT_STRING: C.REQUEST_STRING, C.ROLE_STRING: C.USER_STRING},
        ],
        C.COMPLETION_STRING: [
            {
                C.CONTENT_STRING: completion_content,
                C.ROLE_STRING: C.ASSISTANT_STRING,
            }
        ],
        C.LABEL_STRING: label,
        C.GOAL_STRING: goals,
    }
