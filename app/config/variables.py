import os

# Elasticsearch Configuration
ELASTICSEARCH_PROTOCOL = os.getenv("ELASTICSEARCH_PROTOCOL", "http")
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", 9200))
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "antisemitic")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Data Loading Configuration
DEFAULT_MAX_DOCUMENTS = int(os.getenv("DEFAULT_MAX_DOCUMENTS", 1000))
DEFAULT_SUBSET = os.getenv("DEFAULT_SUBSET", "train")

# API Configuration
MAX_BULK_SIZE = int(os.getenv("MAX_BULK_SIZE", 1000))
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", 10))
MAX_SEARCH_LIMIT = int(os.getenv("MAX_SEARCH_LIMIT", 100))



WEAPONS_PATH = os.getenv("WEAPONS_PATH", "data/weapons.txt")
TWEETS_PATH = os.getenv("TWEETS_PATH", "data/tweets.csv")
MAPPING_PATH = os.getenv("MAPPING_PATH", "config/mapping.json")

SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv("SENTIMENT_THRESHOLD_NEGATIVE", -0.5))
SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv("SENTIMENT_THRESHOLD_POSITIVE", 0.5))
