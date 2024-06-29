"""Vector Database Utilities."""

from langchain_community.vectorstores import Qdrant
from loguru import logger
from qdrant_client import QdrantClient

from utils.ollama import (
    load_embedding_model,
)


def init_qdrant_db(collection_name: str = "default"):
    """Establish a connection to the Qdrant DB."""
    qdrant_client = QdrantClient("http://localhost", port=6333, prefer_grpc=False)
    logger.info(f"USING COLLECTION: {collection_name}")

    embedding = load_embedding_model()
    qdrant = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return qdrant
