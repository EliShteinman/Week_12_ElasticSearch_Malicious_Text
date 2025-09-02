
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator, Iterable

import pandas as pd
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import async_bulk, async_scan
from app.config import variables
from app.models import (DocumentCreate, DocumentResponse, DocumentUpdate,
                        SearchResponse)
logger = logging.getLogger(__name__)


class ElasticSearchRepository:
    def __init__(self, es_client: AsyncElasticsearch, index_name: str = None):
        self.index_name = index_name or variables.ELASTICSEARCH_INDEX_NAME
        self.es = es_client

    # Document operations
    def _build_query(
            self,
            query_text: Optional[str] = None,
            search_terms: Optional[List[str]] = None,
            # Generic filters
            term_filters: Optional[Dict[str, Any]] = None,
            exists_filters: Optional[List[str]] = None,
            not_exists_filters: Optional[List[str]] = None,
            terms_filters: Optional[Dict[str, List[str]]] = None,
            range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
            script_filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generic query builder that supports various filter types.

        Args:
            query_text: Full text search
            search_terms: Terms to search in text field
            term_filters: {field: value} for exact matches
            exists_filters: [field1, field2] for fields that must exist
            not_exists_filters: [field1, field2] for fields that must NOT exist
            terms_filters: {field: [value1, value2]} for multiple values
            range_filters: {field: {"gte": 5, "lt": 10}} for range queries
            script_filters: ["doc['field'].size() >= 2"] for script queries
        """
        must_clauses: List[Dict[str, Any]] = []
        filter_clauses: List[Dict[str, Any]] = []
        must_not_clauses: List[Dict[str, Any]] = []

        # Text search
        if query_text:
            must_clauses.append({'match': {'text': query_text}})
        elif search_terms:
            must_clauses.append({'terms': {'text': search_terms}})
        else:
            must_clauses.append({'match_all': {}})

        # Term filters (exact matches)
        if term_filters:
            for field, value in term_filters.items():
                filter_clauses.append({'term': {field: value}})

        # Exists filters
        if exists_filters:
            for field in exists_filters:
                filter_clauses.append({'exists': {'field': field}})

        # Not exists filters
        if not_exists_filters:
            for field in not_exists_filters:
                must_not_clauses.append({'exists': {'field': field}})

        # Terms filters (multiple values)
        if terms_filters:
            for field, values in terms_filters.items():
                filter_clauses.append({'terms': {field: values}})

        # Range filters
        if range_filters:
            for field, range_config in range_filters.items():
                filter_clauses.append({'range': {field: range_config}})

        # Script filters
        if script_filters:
            for script_source in script_filters:
                filter_clauses.append({
                    'script': {
                        'script': {
                            'source': script_source
                        }
                    }
                })

        return {'bool': {'must': must_clauses, 'filter': filter_clauses, 'must_not': must_not_clauses}}

    async def search_documents(
            self,
            limit: int = 10,
            offset: int = 0,
            **kwargs: Any
    ) -> SearchResponse:
        """
        Performs a paginated search for documents.
        Accepts various filter criteria via kwargs passed to _build_query.
        """
        query = self._build_query(**kwargs)
        search_body = {
            'query': query,
            'from': offset,
            'size': limit,
            'sort': [{'created_at': {'order': 'desc'}}]
        }
        try:
            result = await self.es.search(index=self.index_name, body=search_body)
            documents = [DocumentResponse(id=hit['_id'], **hit['_source']) for hit in result['hits']['hits']]
            return SearchResponse(
                total_hits=result['hits']['total']['value'],
                max_score=result['hits']['max_score'],
                took_ms=result['took'],
                documents=documents
            )
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    # Streaming large result sets
    async def stream_all_documents(
            self,
            fields_to_include: Optional[List[str]] = None,
            **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams all documents matching a query, yielding them one by one.
        Efficient for processing large result sets.
        """
        query = self._build_query(**kwargs)
        search_body = {
            "query": query
        }
        try:
            async for hit in async_scan(
                client=self.es,
                index=self.index_name,
                query=search_body,
                _source=fields_to_include,
                size=200,
            ):
                yield hit
        except Exception as e:
            logger.error(f"Streaming documents failed: {e}", exc_info=True)
            raise



    async def bulk_index_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Efficiently bulk indexes documents from a Pandas DataFrame.
        Uses a generator to avoid loading all data into memory.
        """

        def _generate_actions():
            now = datetime.now(timezone.utc)
            for record in df.to_dict(orient="records"):
                # Ensure all required fields are present if needed, or handle missing ones
                record['created_at'] = now
                record['updated_at'] = now
                yield {"_index": self.index_name, "_source": record}

        try:
            success, failed = await async_bulk(self.es, _generate_actions(), stats_only=True)
            await self.es.indices.refresh(index=self.index_name)
            return {'success_count': success, 'error_count': failed}
        except Exception as e:
            logger.error(f"Bulk indexing from DataFrame failed: {e}")
            raise

    async def bulk_update(self, actions: AsyncGenerator[Dict[str, Any], None]) -> Dict[str, Any]:
        """
        Performs bulk updates using a provided iterable of actions.
        Ideal for enriching documents.
        """
        try:
            success, failed = await async_bulk(self.es, actions, stats_only=True)
            await self.es.indices.refresh(index=self.index_name)
            return {'success_count': success, 'error_count': failed}
        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
            raise


    async def count(self, **kwargs: Any) -> int:
        """Counts documents matching a query."""
        try:
            query = self._build_query(**kwargs)
            response = await self.es.count(index=self.index_name, query=query)
            return response.get('count', 0)
        except Exception as e:
            logger.error(f"Count query failed: {e}", exc_info=True)
            return 0