from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
import secrets
from fastapi import Header, HTTPException

from config import get_settings
from models import (
    ChatRequest, ChatResponse, HealthResponse, MetricsResponse, ErrorResponse
)

from security import SecurityPipeline
from cache import RedisResponseCache
from monitoring import get_logger, MetricsCollector, RequestTimer
from agent import ProductionAgent

import json
from fastapi import Header
from rag import RAGPipeline
from models import IngestRequest, IngestResponse

import asyncio
import io
import secrets
from pathlib import Path

from fastapi import File, Form, Header, HTTPException, UploadFile

load_dotenv()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


security: SecurityPipeline = None
cache: RedisResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
rag: RAGPipeline = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    """

    global security, cache, metrics, agent, rag

    settings = get_settings()

    logger.info(
        "Starting production API...", extra={"extra_data":{
            "enviroment": settings.app_env,
            "primary_model": settings.primary_model,
            "tracing_enabled": settings.langsmith_tracing,
        }}
    )

    # Initialize components

    security = SecurityPipeline()
    cache = RedisResponseCache()
    metrics = MetricsCollector()
    rag = RAGPipeline()
    await rag.setup()
    agent = ProductionAgent(rag)

    logger.info("All components initialized. Ready to serve requests")

    yield

    logger.info("Shutting down...", extra={
        "extra_data": metrics.summary
    })
    await cache.close()
    await rag.close()


# === Rate Limiter Setup ===

limiter = Limiter(key_func=get_remote_address)

# === FastAPI App === 

app = FastAPI(
    title="Production LangGraph API",
    description="A produciton-ready chat API with security, caching and observibility",
    version="1.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter


# === Exception Handler ===

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down."
        }
    )


# === Endpoints ===

@app.get("/")
def home():
    return {"message": "The RAG System Running..."}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Security check (Injection + PII Masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """

    with RequestTimer() as timer:
        security_notes = []

        # -- Step 1: Security Check ---
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={
                "extra_data": {
                    "reason": notes,
                    "thread_id": body.thread_id
                }
            })
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="your message was blocked by our security filters."
            )

        # --- Step 2: Cache Lookup ---
        cached_raw = await cache.get(cleaned_message)
        if cached_raw is not None:
            cached = json.loads(cached_raw)
            metrics.record_request(
                latency_ms=0,
                cache_hit=True
            )
            logger.info("Cache hit", extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                }
            })
            return ChatResponse(
                response=cached["response"],
                sources=cached.get("sources", []),
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0
            )

        # --- Step 3: Invoke LangGrpah Agent ---

        try:
            result = await agent.ainvoke(cleaned_message)
            logger.info(f"agent result keys: {list(result.keys())}")
        except Exception as e:
            logger.error(F"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occured while processing your requests."
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # --- Step 4: Output Validation ---
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # -- Step 5: Cache Store ---
        sources = result["sources"]

        if model_used != "error_handler":
            await cache.set(
                cleaned_message, 
                json.dumps({"response": validated_response, "sources": sources})
            )

        # --- Step 6: Log & Record Metrics ---
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
         logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
         }})

    logger.info("request completed", extra={"extra_data": {
         "thread_id": body.thread_id,
         "model_used": model_used,
         "latency_ms": round(timer.elapsed_ms, 2),
    }})

    return ChatResponse(
        response=validated_response,
        sources=sources,        
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, x_api_key: str = Header(...)):
    if not x_api_key or not secrets.compare_digest(
        x_api_key, get_settings().admin_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")

    ok, reason = security.sanitizer.check(body.text)
    if not ok:
        logger.warning("Suspicious ingest content", extra={"extra_data": {
            "source": body.source,
            "reason": reason
        }})

    count = await rag.ingest(body.text, body.source)
    await cache.invalidate()
    return IngestResponse(source=body.source, chunks_indexed=count)

    



@app.get("/health", response_model=HealthResponse)
async def health():
     """Health check for Docker/Kubernetes"""
     settings = get_settings()

     checks = {
          "agent": agent is not None,
          "security": security is not None,
          "cache": cache is not None,
     }

     all_healthy = all(checks.values())

     return HealthResponse(
          status="healthy" if all_healthy else "degraded",
          environment=settings.app_env,
          checks=checks
     )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
     """Metrics for monitoring dashboards"""
     summary = metrics.summary
     return MetricsResponse(**summary)

@app.get("/cache/stats")
async def cache_stats(): 
    """Cache performance statistics."""
    return await cache.get_stats()






def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = Form(None),
    x_api_key: str = Header(...),
):
    if not secrets.compare_digest(x_api_key, get_settings().admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    filename = Path(file.filename or "upload").name  # strip any path parts
    if Path(filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # PDF/DOCX parsing is CPU-bound and blocking, so keep it off the event loop
    try:
        text = await asyncio.to_thread(extract_text, filename, data)
    except Exception:
        logger.exception("Failed to extract text", extra={"extra_data": {"file": filename}})
        raise HTTPException(status_code=422, detail="Could not read file")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text in file")

    source = source or filename

    ok, reason = security.sanitizer.check(text)
    if not ok:
        logger.warning("Suspicious ingest content", extra={"extra_data": {
            "source": source,
            "reason": reason,
        }})

    count = await rag.ingest(text, source)
    await cache.invalidate()
    return IngestResponse(source=source, chunks_indexed=count)