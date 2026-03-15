from pydantic import BaseModel


class ChunkResponse(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int
    char_count: int
    content: str


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    saved_to: str
    char_count: int
    chunk_size: int
    overlap: int
    total_chunks: int
    preview: str


class DocumentSummary(BaseModel):
    filename: str
    char_count: int
    chunk_size: int
    overlap: int
    total_chunks: int


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[DocumentSummary]


class DocumentDetailResponse(BaseModel):
    filename: str
    saved_to: str
    char_count: int
    chunk_size: int
    overlap: int
    total_chunks: int
    preview: str


class DocumentChunksResponse(BaseModel):
    filename: str
    char_count: int
    chunk_size: int
    overlap: int
    total_chunks: int
    chunks: list[ChunkResponse]


class RetrievalResult(BaseModel):
    filename: str
    chunk_index: int
    start_char: int
    end_char: int
    char_count: int
    score: float
    matched_tokens: list[str]
    content: str


class RetrievalSearchResponse(BaseModel):
    query: str
    top_k: int
    total_hits: int
    results: list[RetrievalResult]


class SourceInfo(BaseModel):
    source_id: int
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    total_sources: int
    sources: list[SourceInfo]
    retrieved_chunks: list[RetrievalResult]