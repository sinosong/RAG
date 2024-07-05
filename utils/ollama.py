from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from loguru import logger


def load_embedding_model(base_url: str = "http://10.6.56.29:11434", model_name: str = "chatfire/bge-m3:q8_0"):
    embeddings = OllamaEmbeddings(
        base_url=base_url, model=model_name
    )
    logger.info("Embedding: Using Ollama")
    return embeddings


def load_llm(base_url: str = "http://10.6.56.29:11434", model_name: str = "qwen2:7b-instruct-q4_0"):
    if len(model_name):
        logger.info(f"LLM: Using Ollama: {model_name}")
        return ChatOllama(
            temperature=0,
            base_url=base_url,
            model=model_name,
            streaming=False,
            # seed=2,
            top_k=10,
            # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,
            # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=10240,  # Sets the size of the context window used to generate the next token.
        )
