from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import time
import traceback
import boto3
from botocore.exceptions import ClientError
from agent import MultiDomainAgent 

# ── Import your existing RAG class ──────────────────────────────────────────
from rag import MultiDomainRAG      # rename your file to rag.py


app = FastAPI(title="MultiDomain RAG API", version="1.0.0")

# ── CORS — allow the React dev server (port 3000 / 5173) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single shared RAG instance (loads embeddings once) ──────────────────────
rag = MultiDomainRAG()
agent = MultiDomainAgent(rag);

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────


class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    query: str
    history: list[HistoryMessage] = []  # Optional conversation history for context

class QueryResponse(BaseModel):
    answer: str
    domain: str
    chunks_used: int
    latency_ms: int
    clarification_question: str | None

class UploadResponse(BaseModel):
    message: str
    files_indexed: list[str]
    latency_ms: int

class UploadFileToS3Response(BaseModel):
    message: str
    files_indexed: list[str]
    latency_ms: int

class StatusResponse(BaseModel):
    status: str
    indexed_domains: list[str]
    total_vectorstores: int

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
def health_check():
    """Check which domains are currently indexed."""
    domains = [str(k) for k in rag.vectorstores.keys()]
    return StatusResponse(
        status="online",
        indexed_domains=domains,
        total_vectorstores=len(domains),
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(file: UploadFile = File(...)):
    reset_index();
    print(f"Received {file} file(s) for upload.")  # Debug log    
    """
    Accept one or more PDF files, save them to the pdfs/ directory,
    then re-run the indexing pipeline so they are immediately searchable.
    """
    for old_file in os.listdir(PDF_DIR):
        if old_file.endswith(".pdf"):
            os.remove(os.path.join(PDF_DIR, old_file))
            print(f"Deleted old file: {old_file}")

    for folder in os.listdir("."):
        if folder.startswith("faiss_") and os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Deleted old FAISS index: {folder}")
    if not file:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_names = []
   # for upload in files:
        # Validate mime type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=415,
            detail=f"{file.filename} is not a PDF.",
        )
    print(f"Saving file: {file.filename}")  # Debug log
    dest = os.path.join(PDF_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    saved_names.append(file.filename)

    # Re-index everything (incremental indexing can be added later)
    start = time.time()
    try:
        rag.documents_by_domain.clear()
        rag.vectorstores.clear()
        rag.retrievers.clear()
        rag.bm25_indexes.clear()      # ← add this
        rag.bm25_chunks.clear()       # ← add this
        rag._save_documents()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    elapsed = int((time.time() - start) * 1000)

    return UploadResponse(
        message=f"Successfully indexed {len(saved_names)} file(s).",
        files_indexed=saved_names,
        latency_ms=elapsed,
    )

@app.post("/api/upload-to-s3")
async def upload_to_s3(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"{file.filename} is not a PDF.")

    s3_client = boto3.client(
        "s3",
        region_name="ap-southeast-2",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="https://s3.ap-southeast-2.amazonaws.com",
        config=boto3.session.Config(
            signature_version="s3v4",
            region_name="ap-southeast-2",
            s3={"addressing_style": "virtual"}
        )
    )

    key = f"ragUploads/{file.filename}"

    try:
        start = time.time()
        s3_client.upload_fileobj(
            file.file,                          # file-like object
            os.getenv("S3_BUCKET_NAME"),        # bucket
            key,                                # s3 key
            ExtraArgs={"ContentType": "application/pdf"}
        )
        elapsed = int((time.time() - start) * 1000)
        status = 'uploaded'
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
        status = 'failed'

    return {
        "message": f"{file.filename} uploaded successfully.",
        "key": key,
        "status": status,
        "latency_ms": elapsed
    }

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    print(f"Received query: {req.query} with history: {req.history}")  # Debug log
    
    """
    Run a RAG query against the indexed vector stores.
    Returns the LLM answer, detected domain, chunk count, and latency.
    """
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not rag.vectorstores:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Please upload PDFs first.",
        )

    start = time.time()
    try:
       # docs = rag.smart_search(req.query)
        domain = rag.detect_query_domain(req.query)
        domain_str = domain.value if domain else "GENERAL"
        print("Agent result:", domain_str)
        history=[h.model_dump() for h in req.history]
        print("Received query from api:", req.history)
        result = agent.run(req.query, domain_str, history=[h.model_dump() for h in req.history])
        print("Agent result:", result)

        # Generate answer via LLM
      #  answer = rag.generate_answer(req.query, domain)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    elapsed = int((time.time() - start) * 1000)
    # Clarification: query was unclear, return the follow-up question
    if result.type == "clarification":
        return QueryResponse(
            answer="",
            domain=domain_str,
            chunks_used=0,
            latency_ms=elapsed,
            used_web_search=False,
            clarification_question=result.answer,
        )

    # Normal answer
    return QueryResponse(
        answer=result.answer,
        domain=domain_str,
        chunks_used=0,
        latency_ms=elapsed,
        used_web_search=result.used_web_search,
        clarification_question=None,
    )


@app.delete("/api/reset")
def reset_index():
    """Clear all in-memory indexes (useful for re-indexing from scratch)."""
    rag.documents_by_domain.clear()
    rag.vectorstores.clear()
    rag.retrievers.clear()
    return {"message": "Index cleared. Upload new PDFs to re-index."}