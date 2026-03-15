# This file defines the in-memory document store and related constants for the application.

from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50
DEFAULT_MODEL = "gpt-5-mini"

DOCUMENT_STORE: dict[str, dict] = {}