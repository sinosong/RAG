"""Vector Database Utilities."""
import os

from langchain_community.vectorstores import Qdrant
from langchain.embeddings.xinference import XinferenceEmbeddings
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from ultra_simple_config import load_config


@load_config(location="config/conf.yml")
def init_qdrant_db(cfg: DictConfig, collection_name: str):
    """Establish a connection to the Qdrant DB."""
    qdrant_client = QdrantClient(cfg.qdrant.url, port=cfg.qdrant.port, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    logger.info(f"USING COLLECTION: {collection_name}")

    embedding = XinferenceEmbeddings(server_url=cfg.embeddings.server_url, model_uid=cfg.embeddings.model_name)

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db
