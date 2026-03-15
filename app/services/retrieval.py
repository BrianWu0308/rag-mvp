import math

from app.services.embeddings import get_single_text_embedding
from app.store import DOCUMENT_STORE


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_chunks(query: str, top_k: int) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    query_embedding = get_single_text_embedding(query)

    scored_results = []

    for doc in DOCUMENT_STORE.values():
        for chunk in doc["chunks"]:
            chunk_embedding = chunk.get("embedding")
            if not chunk_embedding:
                continue

            score = cosine_similarity(query_embedding, chunk_embedding)

            scored_results.append(
                {
                    "filename": doc["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "char_count": chunk["char_count"],
                    "score": score,
                    "matched_tokens": [],
                    "content": chunk["content"],
                }
            )

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    return scored_results[:top_k]