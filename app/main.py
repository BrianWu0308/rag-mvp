from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas.requests import ChatRequest, RetrievalRequest
from app.schemas.responses import (
    ChatResponse,
    DocumentChunksResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
    RetrievalResult,
    RetrievalSearchResponse,
    SourceInfo,
)
from app.services.documents import (
    rebuild_document_store_from_uploads,
    save_uploaded_text_file,
)
from app.services.llm import generate_answer_with_llm
from app.services.retrieval import retrieve_chunks
from app.store import DEFAULT_MODEL, DOCUMENT_STORE


@asynccontextmanager
async def lifespan(app: FastAPI):
    summary = rebuild_document_store_from_uploads()
    print(
        "[startup] document store rebuilt | "
        f"loaded={summary['loaded_count']} | "
        f"from_cache={summary['loaded_from_cache_count']} | "
        f"rebuilt={summary['rebuilt_count']} | "
        f"skipped={summary['skipped_count']}"
    )
    yield


app = FastAPI(title="RAG MVP", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "RAG MVP API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported in MVP stage",
        )

    file_bytes = await file.read()

    try:
        record = save_uploaded_text_file(
            filename=file.filename,
            file_bytes=file_bytes,
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Only UTF-8 encoded .txt files are supported",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {str(e)}",
        )

    return DocumentUploadResponse(
        message="Document uploaded and chunked successfully",
        filename=record["filename"],
        saved_to=record["saved_to"],
        char_count=record["char_count"],
        chunk_size=record["chunk_size"],
        overlap=record["overlap"],
        total_chunks=record["total_chunks"],
        preview=record["content"][:200],
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    documents = [
        DocumentSummary(
            filename=doc["filename"],
            char_count=doc["char_count"],
            chunk_size=doc["chunk_size"],
            overlap=doc["overlap"],
            total_chunks=doc["total_chunks"],
        )
        for doc in DOCUMENT_STORE.values()
    ]

    return DocumentListResponse(
        total_documents=len(documents),
        documents=documents,
    )


@app.get("/documents/{filename}", response_model=DocumentDetailResponse)
def get_document_detail(filename: str):
    doc = DOCUMENT_STORE.get(filename)

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found in memory")

    return DocumentDetailResponse(
        filename=doc["filename"],
        saved_to=doc["saved_to"],
        char_count=doc["char_count"],
        chunk_size=doc["chunk_size"],
        overlap=doc["overlap"],
        total_chunks=doc["total_chunks"],
        preview=doc["content"][:300],
    )


@app.get("/documents/{filename}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(filename: str):
    doc = DOCUMENT_STORE.get(filename)

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found in memory")

    return DocumentChunksResponse(
        filename=doc["filename"],
        char_count=doc["char_count"],
        chunk_size=doc["chunk_size"],
        overlap=doc["overlap"],
        total_chunks=doc["total_chunks"],
        chunks=doc["chunks"],
    )


@app.post("/retrieval/search", response_model=RetrievalSearchResponse)
def retrieval_search(request: RetrievalRequest):
    if not DOCUMENT_STORE:
        raise HTTPException(status_code=400, detail="No documents available in memory")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = retrieve_chunks(query=query, top_k=request.top_k)

    retrieval_results = [
        RetrievalResult(
            filename=item["filename"],
            chunk_index=item["chunk_index"],
            start_char=item["start_char"],
            end_char=item["end_char"],
            char_count=item["char_count"],
            score=item["score"],
            matched_tokens=item["matched_tokens"],
            content=item["content"],
        )
        for item in results
    ]

    return RetrievalSearchResponse(
        query=query,
        top_k=request.top_k,
        total_hits=len(retrieval_results),
        results=retrieval_results,
    )


@app.post("/chat/query", response_model=ChatResponse)
def chat_query(request: ChatRequest):
    if not DOCUMENT_STORE:
        raise HTTPException(status_code=400, detail="No documents available in memory")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrieved_chunks = retrieve_chunks(query=query, top_k=request.top_k)

    retrieval_results = [
        RetrievalResult(
            filename=item["filename"],
            chunk_index=item["chunk_index"],
            start_char=item["start_char"],
            end_char=item["end_char"],
            char_count=item["char_count"],
            score=item["score"],
            matched_tokens=item["matched_tokens"],
            content=item["content"],
        )
        for item in retrieved_chunks
    ]

    if not retrieval_results:
        return ChatResponse(
            query=query,
            answer="我在目前上傳的文件中找不到足夠相關的內容，所以無法可靠回答。",
            model=DEFAULT_MODEL,
            total_sources=0,
            sources=[],
            retrieved_chunks=[],
        )

    try:
        answer = generate_answer_with_llm(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    sources = [
        SourceInfo(
            source_id=idx,
            filename=chunk["filename"],
            chunk_index=chunk["chunk_index"],
            score=chunk["score"],
        )
        for idx, chunk in enumerate(retrieved_chunks, start=1)
    ]

    return ChatResponse(
        query=query,
        answer=answer,
        model=DEFAULT_MODEL,
        total_sources=len(sources),
        sources=sources,
        retrieved_chunks=retrieval_results,
    )