from elasticsearch import AsyncElasticsearch
from typing import Optional


_es_client: Optional[AsyncElasticsearch] = None

def set_es_client(client: AsyncElasticsearch) -> None:
    global _es_client
    _es_client = client

def get_es_client() -> AsyncElasticsearch:
    if _es_client is None:
        raise ValueError("Elasticsearch client is not initialized.")
    return _es_client

def is_client_ready() -> bool:
    return _es_client is not None


def cleanup_resources() -> None:
    global _es_client
    _es_client = None
