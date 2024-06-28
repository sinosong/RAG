"""Local Backend Service."""
import glob
import os
from typing import Optional

from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from loguru import logger

from utils.vdb import init_qdrant_db


def get_db_connection(collection_name: str) -> Qdrant:
    """Initializes a connection to the Qdrant DB.
    Args:
        :param collection_name:
    Returns:
        Qdrant: The Qdrant DB connection.
    """
    if collection_name is None or not collection_name:
        collection_name = "default"

    return init_qdrant_db(collection_name)


def embedd_documents(segmented_files_dir: str, collection_name: Optional[str] = None) -> None:
    """将文本段写入vb
    """
    vector_db: Qdrant = get_db_connection(collection_name=collection_name)

    segmented_files = glob.glob(os.path.join(segmented_files_dir, "*.log"))
    docs = []
    for file_path in segmented_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append(Document(page_content=content))

    logger.info(f"Loaded {len(docs)} documents.")
    texts = [doc.page_content for doc in docs]
    print(texts)
    """ todo 确认meta_datas存储的数据标准，至少包括：代码分支、接口、路径、文件名等信息
    文件路径：记录源代码文件在项目中的相对或绝对路径，这有助于在检索时知道代码片段来自哪个具体文件。
    代码行号或位置：标识代码片段在文件中的起始和结束位置，这对于调试和维护非常重要。
    文件类型或编程语言：说明文件是用哪种编程语言编写的，如Python、Java或C++，这有助于进行特定语言的查询或分析。
    作者或提交者：如果是从版本控制系统获取的源代码，metadata可能包含提交代码的作者或提交者信息，以及提交日期。
    版本控制信息：如Git的commit ID，这可以追踪代码的历史版本和变更。
    代码注释或文档字符串：如果有的话，这些可以作为额外的上下文，解释代码的功能或意图。
    项目名称或模块名称：帮助理解代码所属的项目或模块，这对于大型软件工程尤其重要。
    依赖关系：列出代码片段可能依赖的外部库或模块，这对于构建环境或理解代码运行上下文是有价值的。
    功能标签或分类：标记代码属于哪一类功能，如“用户认证”、“日志记录”等，便于根据功能进行搜索。
    """
    meta_datas = [doc.metadata for doc in docs]
    print(meta_datas)
    # vector_db.add_texts(texts=texts, metadatas=meta_datas)
    logger.info("SUCCESS: Texts embedded.")
