from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [text.strip() for text in texts if text.strip()]

    if not cleaned_texts:
        raise ValueError("No valid texts to embed")

    client = OpenAI()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_texts,
    )

    return [item.embedding for item in response.data]


def get_single_text_embedding(text: str) -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError("Text to embed cannot be empty")

    embeddings = get_text_embeddings([text])
    return embeddings[0]