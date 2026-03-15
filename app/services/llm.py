from openai import OpenAI

from app.store import DEFAULT_MODEL


def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {idx}]\n"
            f"filename: {chunk['filename']}\n"
            f"chunk_index: {chunk['chunk_index']}\n"
            f"score: {chunk['score']}\n"
            f"content:\n{chunk['content']}"
        )

    return "\n\n".join(context_parts)


def generate_answer_with_llm(query: str, retrieved_chunks: list[dict]) -> str:
    context_text = build_context(retrieved_chunks)

    system_prompt = (
        "You are a helpful RAG QA assistant. "
        "Answer only based on the provided context. "
        "Do not use outside knowledge. "
        "If the context is insufficient, say you do not know. "
        "When possible, cite sources using [Source 1], [Source 2], etc. "
        "Respond in Traditional Chinese."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Please answer the question using only the context above."
    )

    client = OpenAI()

    response = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )

    answer = response.output_text.strip()

    if not answer:
        return "模型沒有回傳可用文字內容。"

    return answer