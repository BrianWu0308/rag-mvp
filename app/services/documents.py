from pathlib import Path

from app.services.cache import load_cached_document_if_valid, save_document_cache
from app.services.chunking import split_text_into_chunks
from app.services.embeddings import get_text_embeddings
from app.store import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DOCUMENT_STORE,
    UPLOAD_DIR,
)


def build_document_record(filename: str, text_content: str, save_path: Path) -> dict:
    chunks = split_text_into_chunks(
        text=text_content,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )

    chunk_texts = [chunk["content"] for chunk in chunks]
    embeddings = get_text_embeddings(chunk_texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return {
        "filename": filename,
        "saved_to": str(save_path),
        "content": text_content,
        "char_count": len(text_content),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


def save_uploaded_text_file(filename: str, file_bytes: bytes) -> dict:
    text_content = file_bytes.decode("utf-8")

    save_path = UPLOAD_DIR / filename
    save_path.write_bytes(file_bytes)

    record = build_document_record(
        filename=filename,
        text_content=text_content,
        save_path=save_path,
    )

    DOCUMENT_STORE[filename] = record
    save_document_cache(record)

    return record


def rebuild_document_store_from_uploads() -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)

    DOCUMENT_STORE.clear()

    loaded_files = []
    loaded_from_cache = []
    rebuilt_files = []
    skipped_files = []

    for file_path in sorted(UPLOAD_DIR.glob("*.txt")):
        cached_record = load_cached_document_if_valid(file_path)

        if cached_record is not None:
            DOCUMENT_STORE[file_path.name] = cached_record
            loaded_files.append(file_path.name)
            loaded_from_cache.append(file_path.name)
            continue

        try:
            text_content = file_path.read_text(encoding="utf-8")

            record = build_document_record(
                filename=file_path.name,
                text_content=text_content,
                save_path=file_path,
            )

            DOCUMENT_STORE[file_path.name] = record
            save_document_cache(record)

            loaded_files.append(file_path.name)
            rebuilt_files.append(file_path.name)

        except UnicodeDecodeError:
            skipped_files.append(file_path.name)
        except Exception:
            skipped_files.append(file_path.name)

    return {
        "loaded_count": len(loaded_files),
        "loaded_files": loaded_files,
        "loaded_from_cache_count": len(loaded_from_cache),
        "loaded_from_cache_files": loaded_from_cache,
        "rebuilt_count": len(rebuilt_files),
        "rebuilt_files": rebuilt_files,
        "skipped_count": len(skipped_files),
        "skipped_files": skipped_files,
    }