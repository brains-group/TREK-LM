# Description: This file contains constants used across the project.

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

# Dataset keys
PROMPT_STRING = "prompt"
COMPLETION_STRING = "completion"
LABEL_STRING = "label"
GOAL_STRING = "goal"

# Entity types
USER_TYPE = "User"
MOVIE_TYPE = "Movie"
USER_ID_FORMAT = "http://example.org/user/{}"
ID_STRING = "@id"
TYPE_STRING = "@type"
NAME_STRING = "name"

# Prompt templates
KG_PREFACE_STRING = "You perform movie recommendations based on a Knowledge Graph. You will recommend a list of movies to the user that are not already in their knowledge graph. The user's entity is represented by {}. Use this knowledge graph when responding to their queries: {}. Provide the recommendations as a bulleted list with dashes (-) as the bullet points."
REQUEST_STRING = "Recommend movies to me."
COMPLETION_FORMAT_STRING = "Based on your Knowledge Graph, I recommend the following:\n{}"
