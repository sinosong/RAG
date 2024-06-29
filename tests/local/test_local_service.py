"""Test the service."""
import os
import pytest
from loguru import logger

from backend.local_service import (
    embedd_documents,
)

root_path = "/Users/sinosong/project/public3rd/RAG"


def test_embedd_documents() -> None:
    """Test that embedd_documents does not raise an error."""
    os.chdir(root_path)
    embedd_documents("tests/resources/", "express")
