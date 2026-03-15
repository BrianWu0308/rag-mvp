import json
from pathlib import Path

from app.store import CACHE_DIR

CACHE_VERSION = 1


def get_cache_path(filename: str) -> Path:
    safe_name = f"{Path(filename).name}.json"
    return CACHE_DIR / safe_name


def save_document_cache(record: dict) -> None:
    source_path = Path(record["saved_to"])
    stat = source_path.stat()

    payload = {
        "cache_version": CACHE_VERSION,
        "source_filename": record["filename"],
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size_bytes": stat.st_size,
        "record": record,
    }

    cache_path = get_cache_path(record["filename"])
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def load_cached_document_if_valid(file_path: Path) -> dict | None:
    cache_path = get_cache_path(file_path.name)

    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None

    try:
        stat = file_path.stat()
    except FileNotFoundError:
        return None

    cached_mtime_ns = payload.get("source_mtime_ns")
    cached_size_bytes = payload.get("source_size_bytes")

    if cached_mtime_ns != stat.st_mtime_ns:
        return None

    if cached_size_bytes != stat.st_size:
        return None

    record = payload.get("record")
    if not isinstance(record, dict):
        return None

    record["saved_to"] = str(file_path)
    record["filename"] = file_path.name

    return record