"""
This module provides utility functions for creating RDF URIs consistently across the application.
"""

from rdflib import URIRef
from utils import constants as C


def get_user_uri(user_id):
    """Creates a URIRef for a user."""
    return URIRef(f"{C.EX}user/{user_id}")


def get_movie_uri(movie_name):
    """Creates a URIRef for a movie."""
    return URIRef(f"{C.EX}movie/{movie_name.replace(' ', '_')}")


def get_relation_uri(relation_name):
    """Creates a URIRef for a relation."""
    return URIRef(f"{C.REL}{relation_name}")
