"""
A small library of curated Personal Knowledge Graphs (PKGs) for the FedTREK-LM
demo. Each PKG captures a distinct user taste profile as a set of liked and
disliked movies.

The PKGs are serialized into exactly the same JSON-LD + system-prompt format
used to train and evaluate the model (see ``utils/kg_creation.py`` and
``utils/constants.py``), so the model sees inputs that are indistinguishable
from those used in the original ESWC paper.
"""

from rdflib import Graph, Literal
from rdflib.namespace import RDF, RDFS

from utils import constants as C
from utils.kg_creation import preface_turn
from utils.uri_helpers import get_user_uri, get_movie_uri, get_relation_uri


# ---------------------------------------------------------------------------
# Curated taste profiles.
#
# Each entry describes one selectable user. ``liked`` / ``disliked`` are lists
# of movie titles written in the same "Title (Year)" style as the ReDial-derived
# training data. ``user_id`` mirrors the integer client identifiers used in the
# Movie PKG dataset.
# ---------------------------------------------------------------------------
PKG_LIBRARY = {
    "action": {
        "name": "Action & Thriller Fan",
        "user_id": "101",
        "description": "Enjoys high-octane action and crime thrillers.",
        "liked": [
            "Die Hard (1988)",
            "Mad Max: Fury Road (2015)",
            "John Wick (2014)",
            "The Dark Knight (2008)",
            "Heat (1995)",
            "Mission: Impossible - Fallout (2018)",
        ],
        "disliked": [
            "The Notebook (2004)",
        ],
    },
    "scifi": {
        "name": "Science-Fiction Fan",
        "user_id": "102",
        "description": "Loves cerebral and space-faring science fiction.",
        "liked": [
            "Blade Runner 2049 (2017)",
            "Interstellar (2014)",
            "Arrival (2016)",
            "The Matrix (1999)",
            "Inception (2010)",
            "Ex Machina (2014)",
        ],
        "disliked": [
            "Bridesmaids (2011)",
        ],
    },
    "animation": {
        "name": "Animation & Family Fan",
        "user_id": "103",
        "description": "Prefers animated features and feel-good family films.",
        "liked": [
            "Toy Story (1995)",
            "Spirited Away (2001)",
            "Finding Nemo (2003)",
            "The Incredibles (2004)",
            "Coco (2017)",
            "Up (2009)",
        ],
        "disliked": [
            "Saw (2004)",
        ],
    },
    "horror": {
        "name": "Horror Fan",
        "user_id": "104",
        "description": "Seeks out suspenseful horror and supernatural thrillers.",
        "liked": [
            "The Conjuring (2013)",
            "Hereditary (2018)",
            "Get Out (2017)",
            "A Quiet Place (2018)",
            "The Shining (1980)",
            "It (2017)",
        ],
        "disliked": [
            "Toy Story (1995)",
        ],
    },
    "romcom": {
        "name": "Romantic Comedy Fan",
        "user_id": "105",
        "description": "Likes lighthearted romantic comedies and dramas.",
        "liked": [
            "When Harry Met Sally (1989)",
            "Pretty Woman (1990)",
            "Notting Hill (1999)",
            "Crazy Rich Asians (2018)",
            "La La Land (2016)",
            "Love Actually (2003)",
        ],
        "disliked": [
            "The Texas Chain Saw Massacre (1974)",
        ],
    },
    "classic": {
        "name": "Classic Drama Fan",
        "user_id": "106",
        "description": "Appreciates acclaimed classic and prestige dramas.",
        "liked": [
            "The Godfather (1972)",
            "Schindler's List (1993)",
            "12 Angry Men (1957)",
            "Casablanca (1942)",
            "Forrest Gump (1994)",
            "The Shawshank Redemption (1994)",
        ],
        "disliked": [
            "Transformers (2007)",
        ],
    },
}


def build_graph(profile):
    """Builds an rdflib Graph for a taste profile, matching the training format.

    Args:
        profile (dict): An entry from ``PKG_LIBRARY``.

    Returns:
        tuple: ``(graph, user_uri)`` where ``graph`` is the populated rdflib
        Graph and ``user_uri`` is the user's URIRef.
    """
    g = Graph()
    user_uri = get_user_uri(profile["user_id"])
    g.add((user_uri, RDF.type, C.USER_TYPE))

    def _add(movie_name, relation):
        movie_uri = get_movie_uri(movie_name)
        g.add((movie_uri, RDF.type, C.MOVIE_TYPE))
        g.add((movie_uri, RDFS.label, Literal(movie_name)))
        g.add((user_uri, get_relation_uri(relation), movie_uri))

    for movie_name in profile.get("liked", []):
        _add(movie_name, C.LIKED_STRING)
    for movie_name in profile.get("disliked", []):
        _add(movie_name, C.DISLIKED_STRING)

    return g, user_uri


def build_system_prompt(profile):
    """Returns the full system prompt (KG preface + serialized JSON-LD) for a profile."""
    g, user_uri = build_graph(profile)
    return preface_turn(str(user_uri), g)


def get_profile(key):
    """Looks up a profile by its library key."""
    return PKG_LIBRARY[key]


def list_profiles():
    """Returns ``(key, display_name)`` pairs for populating a selector."""
    return [(key, prof["name"]) for key, prof in PKG_LIBRARY.items()]
