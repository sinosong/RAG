from backend.local_service import (
    embedd_documents,
)
from utils.vdb import (
    init_qdrant_db,
)

if __name__ == '__main__':
    init_qdrant_db("test")
