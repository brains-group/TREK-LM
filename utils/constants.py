# Description: This file contains constants used across the project.
from rdflib import URIRef

# Role identifiers
USER_STRING = "user"
ASSISTANT_STRING = "assistant"
SYSTEM_STRING = "system"

# Message structure
CONTENT_STRING = "content"
ROLE_STRING = "role"

# Knowledge Graph structure
HEAD_STRING = "head"
RELATION_STRING = "relation"
TAIL_STRING = "tail"

# Relation types
LIKED_STRING = "liked"
DISLIKED_STRING = "disliked"
SEEN_STRING = "seen"
UNSEEN_STRING = "unseen"
SUGGESTED_STRING = "suggested"

# Food-specific relation types
NAME_STRING = URIRef("name")
MINUTES_STRING = URIRef("minutesToCook")
AUTHOR_ID_STRING = URIRef("authorID")
SUBMISSION_DATE_STRING = URIRef("uploadDate")
HAS_TAG_STRING = URIRef("hasTag")
CALORIES_STRING = URIRef("numCalories")
TOTAL_FAT_STRING = URIRef("totalFatPercentDailyValue")
SUGAR_STRING = URIRef("sugarPercentDailyValue")
SODIUM_STRING = URIRef("sodiumPercentDailyValue")
PROTEIN_STRING = URIRef("proteinPercentDailyValue")
SAT_FAT_STRING = URIRef("saturatedFatPercentDailyValue")
CARBS_STRING = URIRef("carbohydratesPercentDailyValue")
STEPS_STRING = URIRef("numRecipeSteps")
HAS_INGREDIENT_STRING = URIRef("hasIngredient")
RELATION_RATING_PREFIX = "rating"  # e.g., rating_0, rating_1, etc.

# Dataset keys
PROMPT_STRING = "prompt"
COMPLETION_STRING = "completion"
LABEL_STRING = "label"
GOAL_STRING = "goal"

# Prompt templates
KG_PREFACE_STRING = "You perform movie recommendations based on a Knowledge Graph. You will recommend a list of movies to the user that are not already in their knowledge graph. The user's entity is represented by {}. Use this knowledge graph when responding to their queries: {}. Provide the recommendations as a bulleted list with dashes (-) as the bullet points."
REQUEST_STRING = "Recommend movies to me."
COMPLETION_FORMAT_STRING = (
    "Based on your Knowledge Graph, I recommend the following:\n{}"
)

# Food-specific prompt templates
FOOD_KG_PREFACE_STRING = "You perform recipe recommendations based on a Knowledge Graph. You will recommend a list of recipes to the user that are not already in their knowledge graph. The user's entity is represented by {}. Use this knowledge graph when responding to their queries: {}. Provide the recommendations as a bulleted list with dashes (-) as the bullet points."
FOOD_REQUEST_STRING = "Recommend recipes to me."
FOOD_COMPLETION_FORMAT_STRING = (
    "Based on your Knowledge Graph, I recommend the following:\n{}"
)

# RDF Namespaces
EX = "http://movie-recs.org/"
REL = "http://movie-recs.org/relations/"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
MOVIE_PREFIX = f"{EX}movie/"
USER_PREFIX = f"{EX}user/"

# Food-specific RDF Namespaces
RECIPE_PREFIX = f"{EX}recipe/"

# Entity types
USER_TYPE = URIRef(f"{EX}User")
MOVIE_TYPE = URIRef(f"{EX}Movie")

# Food-specific entity types
RECIPE_TYPE = URIRef(f"{EX}Recipe")
