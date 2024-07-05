from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.vector_database_service import (
    embedd_local_segmented_files,
)
from utils.vdb import (
    init_qdrant_db,
)

if __name__ == '__main__':
    init_qdrant_db("test")
