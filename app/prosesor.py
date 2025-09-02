import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import pandas as pd

from app.config import variables
from app.dal.data_loader import DataLoader
from app.dependencies.elasticsearch import get_es_client
from app.utils.elasticSearch_repository import ElasticSearchRepository
from app.utils.sentiment_analyzer import SentimentAnalyzer
from app.utils.weapon_detector import WeaponDetector

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.es_client = get_es_client()
        self.data_loader = DataLoader()
        self.sentiment_analyzer = SentimentAnalyzer()
        index_name = variables.ELASTICSEARCH_INDEX_NAME
        self.es_repository = ElasticSearchRepository(self.es_client, index_name)

    async def process(self):
        await self._load_and_index_tweets()
        await self.es_repository.refresh()
        await self._enrich_documents_with_emotion()
        await self.es_repository.refresh()
        await self._enrich_documents_with_weapon_info()
        await self.es_repository.refresh()
        await self._cleanup_irrelevant_documents()
        await self.es_repository.refresh()

    async def _load_and_index_tweets(self):
        tweets_df = self.data_loader.load_data(variables.TWEETS_PATH)
        cleaned_df = self._validate_and_clean_dataframe(tweets_df)
        result = await self.es_repository.bulk_index_from_dataframe(cleaned_df)
        return result

    def _validate_and_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df_clean = df.copy()
        df_clean["CreateDate"] = pd.to_datetime(
            df_clean.get("CreateDate"), errors="coerce"
        )
        df_clean["TweetID"] = pd.to_numeric(df_clean.get("TweetID"), errors="coerce")
        df_clean.dropna(subset=["CreateDate", "TweetID", "text"], inplace=True)
        df_clean["TweetID"] = df_clean["TweetID"].astype(float)
        df_clean["Antisemitic"] = df_clean["Antisemitic"].astype(bool)
        return df_clean

    async def _generic_enrich_documents(
        self,
        field_name: str,
        analyzer_func: Callable[[str], Any],
        search_params: Dict[str, Any],
        process_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generic method for enriching documents with analyzed data.

        Args:
            field_name: Name of the field to add/update
            analyzer_func: Function that analyzes text and returns result
            search_params: Parameters for filtering documents to process
            process_name: Name for logging purposes
        """
        docs_to_process = await self.es_repository.count(**search_params)
        if docs_to_process == 0:
            logger.info(f"No documents to process for {process_name}")
            return None

        logger.info(f"Processing {docs_to_process} documents for {process_name}")

        async def generate_update_actions() -> AsyncGenerator[Dict[str, Any], None]:
            stream = self.es_repository.stream_all_documents(
                fields_to_include=["text"], **search_params
            )

            processed_count = 0
            async for doc in stream:
                text_to_analyze = doc["_source"].get("text")
                if not text_to_analyze:
                    continue

                analyzed_result = analyzer_func(text_to_analyze)
                if analyzed_result:
                    yield {
                        "_op_type": "update",
                        "_index": self.es_repository.index_name,
                        "_id": doc["_id"],
                        "doc": {
                            field_name: analyzed_result,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    }
                    processed_count += 1

                    if processed_count % 100 == 0:  # Log progress
                        logger.info(
                            f"Processed {processed_count} documents for {process_name}"
                        )

        result = await self.es_repository.bulk_update(generate_update_actions())
        logger.info(f"Completed {process_name}: {result}")
        return result

    async def _enrich_documents_with_emotion(self):
        """Enrich documents with emotion analysis."""
        return await self._generic_enrich_documents(
            field_name="emotion",
            analyzer_func=self.sentiment_analyzer.get_sentiment_label,
            search_params={"not_exists_filters": ["emotion"]},
            process_name="emotion enrichment",
        )

    async def _enrich_documents_with_weapon_info(self):
        """Enrich documents with weapon detection."""
        weapon_as_list = self.data_loader.load_lines_as_list(variables.WEAPONS_PATH)
        weapon_detector = WeaponDetector(weapon_as_list)

        return await self._generic_enrich_documents(
            field_name="weapons",
            analyzer_func=weapon_detector.find_weapons,
            search_params={"terms_filters": {"text": weapon_as_list}},
            process_name="weapon detection",
        )

    async def _cleanup_irrelevant_documents(self):
        search_params = {
            "term_filters": {"Antisemitic": False},
            "not_exists_filters": ["weapons"],
            "terms_filters": {"emotion": ["neutral", "positive"]},
        }

        docs_to_delete = await self.es_repository.count(**search_params)
        if docs_to_delete == 0:
            logger.info("No irrelevant documents found to delete")
            return None

        logger.info(f"Deleting {docs_to_delete} irrelevant documents")

        async def generate_delete_actions() -> AsyncGenerator[Dict[str, Any], None]:
            stream = self.es_repository.stream_all_documents(
                fields_to_include=[], **search_params
            )

            delete_count = 0
            async for doc in stream:
                yield {
                    "_op_type": "delete",
                    "_index": self.es_repository.index_name,
                    "_id": doc["_id"],
                }
                delete_count += 1

                if delete_count % 100 == 0:
                    logger.info(f"Marked {delete_count} documents for deletion")

        result = await self.es_repository.bulk_update(generate_delete_actions())
        logger.info(f"Cleanup completed: {result}")
        return result
