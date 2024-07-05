from langchain.embeddings.xinference import XinferenceEmbeddings
from loguru import logger


def load_embedding_model(server_url: str = "http://10.6.56.28:9997", model_uid: str = "bce-embedding-base_v1"):
    embeddings = XinferenceEmbeddings(server_url=server_url, model_uid=model_uid)
    logger.info("Embedding: Using xinference")
    return embeddings
