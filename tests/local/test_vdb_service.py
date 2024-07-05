"""Test the service."""
import os
import pytest
from loguru import logger

from backend.vector_database_service import (
    embedd_local_segmented_files,
)

root_path = "/Users/esvc/biyao/tool/RAG"


def test_embedd_documents() -> None:
    """Test that embedd_documents does not raise an error."""
    os.chdir(root_path)
    embedd_local_segmented_files("tests/resources/", "express")
