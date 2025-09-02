import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import variables
from app.dal.data_loader import DataLoader
from app.dal.elasticsearch import ElasticsearchCoon
from app.dependencies.elasticsearch import (
    cleanup_resources,
    get_es_client,
    set_es_client,
)
from app.prosesor import DataProcessor
from app.utils.elasticSearch_repository import ElasticSearchRepository

logging.basicConfig(
    level=getattr(logging, variables.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan startup")

    try:
        data_loader = DataLoader()
        es_url = f"{variables.ELASTICSEARCH_PROTOCOL}://{variables.ELASTICSEARCH_HOST}:{variables.ELASTICSEARCH_PORT}/"
        logger.info(f"Initializing Elasticsearch client with URL: {es_url}")
        mapping = data_loader.load_mapping(variables.MAPPING_PATH)
        logger.info(f"Loaded mapping from {variables.MAPPING_PATH}")
        logger.debug(f"Mapping: {mapping}")
        index_name = variables.ELASTICSEARCH_INDEX_NAME
        logger.info(f"Using Elasticsearch index name: {index_name}")

        es_client = ElasticsearchCoon(es_url, index_name, mapping)
        set_es_client(es_client.get_es_client())
        await es_client.initialize_index()

        processor = DataProcessor()
        asyncio.create_task(processor.process())
        logger.info("Background data processing task started")

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

    yield

    # Shutdown - use dependency to get client
    try:
        es = get_es_client()
        await es.close()
        cleanup_resources()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="20 Newsgroups Search API",
    description="A full CRUD API for newsgroup documents with Elasticsearch backend",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "20 Newsgroups Search API",
        "version": "2.0.0",
        "description": "CRUD API for 20newsgroups dataset with Elasticsearch",
        "endpoints": {
            "docs": "/docs",
        },
    }


@app.get("/api/antisemitic-with-weapons", tags=["search"])
async def get_antisemitic_with_weapons():
    """
    Returns all antisemitic documents that have weapons.
    Only works if processing is complete.
    """
    try:
        es_client = get_es_client()
        es_repository = ElasticSearchRepository(
            es_client, variables.ELASTICSEARCH_INDEX_NAME
        )

        # Check if processing is complete
        missing_emotion = await es_repository.count(not_exists_filters=["emotion"])
        total_docs = await es_repository.count()

        if total_docs == 0 or missing_emotion > 0:
            return {
                "status": "processing_incomplete",
                "message": "Document processing is not yet complete. Please try again later.",
                "documents": [],
            }

        # Search for antisemitic documents with weapons
        result = await es_repository.search_documents(
            limit=10000, term_filters={"Antisemitic": True}, exists_filters=["weapons"]
        )

        return {
            "status": "success",
            "message": f"Found {len(result.documents)} antisemitic documents with weapons",
            "total_count": result.total_hits,
            "documents": result.documents,
        }

    except Exception as e:
        logger.error(f"Error in antisemitic-with-weapons endpoint: {e}")
        return {"status": "error", "message": "Internal server error", "documents": []}


@app.get("/api/multiple-weapons", tags=["search"])
async def get_multiple_weapons():
    """
    Returns all documents that have 2 or more weapons.
    Only works if processing is complete.
    """
    try:
        es_client = get_es_client()
        es_repository = ElasticSearchRepository(
            es_client, variables.ELASTICSEARCH_INDEX_NAME
        )

        # Check if processing is complete
        missing_emotion = await es_repository.count(not_exists_filters=["emotion"])
        total_docs = await es_repository.count()

        if total_docs == 0 or missing_emotion > 0:
            return {
                "status": "processing_incomplete",
                "message": "Document processing is not yet complete. Please try again later.",
                "documents": [],
            }

        # Search for documents with 2+ weapons
        result = await es_repository.search_documents(
            limit=10000, script_filters=["doc['weapons'].size() >= 2"]
        )

        return {
            "status": "success",
            "message": f"Found {len(result.documents)} documents with 2+ weapons",
            "total_count": result.total_hits,
            "documents": result.documents,
        }

    except Exception as e:
        logger.error(f"Error in multiple-weapons endpoint: {e}")
        return {"status": "error", "message": "Internal server error", "documents": []}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8182)
