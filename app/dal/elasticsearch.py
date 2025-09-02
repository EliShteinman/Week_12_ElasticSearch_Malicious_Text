import logging
from typing import Any, Dict

from elasticsearch import AsyncElasticsearch


logger = logging.getLogger(__name__)

class ElasticsearchCoon:
    def __init__(self,es_url: str, index_name: str = None, mapping: Dict[str, Any] = None):
        self.es = AsyncElasticsearch(es_url)
        self.index_name = index_name
        self.mapping = mapping

    async def initialize_index(self, index_name: str = None, mapping: Dict[str, Any] = None) -> None:
        """Initialize the Elasticsearch index with the default mapping."""
        index_name = index_name or self.index_name
        if not index_name:
            raise ValueError("Index name must be provided either during initialization or as a parameter.")
        mapping = mapping or self.mapping
        if not mapping:
            logger.warning("No mapping provided. Using default mapping.")
        try:
            await self.es.indices.delete(index=index_name, ignore_unavailable=True)
            logger.info(f"Deleted index '{index_name}' if it existed.")
            if mapping:
                await self.es.indices.create(index=index_name, mappings=mapping)
            else:
                logger.warning("No mapping provided. Using default mapping.")
                await self.es.indices.create(index=index_name)
            logger.info(f"Created index '{index_name}' with mapping.")
            await self.es.indices.refresh(index=index_name)
        except Exception as e:
            logger.error(f"Failed to initialize index '{index_name}': {e}")
            raise

    def get_es_client(self) -> AsyncElasticsearch:
        return self.es