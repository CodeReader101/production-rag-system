from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import ProductionAgent
from cache import RedisResponseCache
from config import get_settings
from models import (
    ChatRequest, ChatResponse, HealthResponse, IngestRequest,
    IngestResponse, MetricsResponse,
)
from monitoring import MetricsCollector, RequestTimer, get_logger
from rag import RAGPipeline
from security import SecurityPipeline

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
    global security, cache, metrics, agent, rag

    settings = get_settings()
    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langsmith_tracing,
    }})

    security = SecurityPipeline()
    cache = RedisResponseCache(ttl_seconds=settings.cache_ttl_seconds,
                               url=settings.redis_url)
    metrics = MetricsCollector()
    rag = RAGPipeline()
    await rag.setup()
    agent = ProductionAgent(rag)

    logger.info("All components initialized. Ready to serve requests")
    yield

    logger.info("Shutting down...", extra={"extra_data": metrics.summary})
    await cache.close()
    await rag.close()


# === App + rate limiter ===

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Production LangGraph RAG API",
    description="A production-ready RAG chat API with security, caching and observability",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded",
                 "detail": "Too many requests. Please slow down."},
    )


# === Auth ===

def require_admin(x_api_key: str | None = Header(default=None)):
    expected = get_settings().admin_api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _parse_cached(raw: str | None) -> dict | None:
    """Cache entries are JSON; anything else counts as a miss."""
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return {"response": data["response"], "sources": data.get("sources", [])}
    except (ValueError, KeyError, TypeError):
        return None


# === Endpoints ===

@app.get("/")
def home():
    return {"message": "The RAG System Running..."}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
async def chat(request: Request, body: ChatRequest):
    """
    Flow: security check -> cache lookup -> retrieve + generate -> output validation
    -> cache store -> response.
    No @traceable here: this function sees the raw, unmasked message.
    """
    with RequestTimer() as timer:
        security_notes = []

        # Step 1: security check
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes, "thread_id": body.thread_id}})
            metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
            raise HTTPException(status_code=400,
                                detail="Your message was blocked by our security filters.")

        if not cleaned_message.strip():
            metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
            raise HTTPException(status_code=400,
                                detail="Message is empty after sanitization.")

        # Step 2: cache lookup
        cached = _parse_cached(await cache.get(cleaned_message))
        if cached is not None:
            metrics.record_request(latency_ms=timer.elapsed_ms, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {"thread_id": body.thread_id}})
            return ChatResponse(
                response=cached["response"],
                sources=cached["sources"],
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=round(timer.elapsed_ms, 2),
            )

        # Step 3: retrieve + generate
        try:
            result = await agent.ainvoke(cleaned_message)
        except Exception:
            logger.exception("Agent invocation failed",
                             extra={"extra_data": {"thread_id": body.thread_id}})
            metrics.record_request(latency_ms=timer.elapsed_ms, error=True)
            raise HTTPException(status_code=500,
                                detail="An error occurred while processing your request.")

        response_text = result["response"]
        model_used = result["model_used"]
        sources = result.get("sources", [])
        context = result.get("context", "")

        # Step 4: output validation
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # Step 5: cache store (not for error or blocked responses)
        if model_used != "error_handler" and "Harmful content blocked" not in output_warnings:
            await cache.set(cleaned_message, json.dumps(
                {"response": validated_response, "sources": sources}))

    # Step 6: log and record metrics
    input_tokens = int((len(cleaned_message.split()) + len(context.split())) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes, "thread_id": body.thread_id}})
    logger.info("Request completed", extra={"extra_data": {
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


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest(body: IngestRequest):
    """Index raw text. Re-ingesting the same source replaces its chunks."""
    ok, reason = security.sanitizer.check(body.text)
    if not ok:
        # Logged, not rejected: patterns like "system prompt" appear in normal documents
        logger.warning("Suspicious ingest content", extra={"extra_data": {
            "source": body.source, "reason": reason}})

    count = await rag.ingest(body.text, body.source)
    await cache.invalidate()
    return IngestResponse(source=body.source, chunks_indexed=count)


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


@app.post("/ingest/file", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = Form(None),
):
    """Upload a .txt, .md, .pdf or .docx. `source` defaults to the filename without extension."""
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

    source = source or Path(filename).stem
    if len(source) > 300:
        raise HTTPException(status_code=400, detail="source is too long (max 300 characters)")

    ok, reason = security.sanitizer.check(text)
    if not ok:
        logger.warning("Suspicious ingest content", extra={"extra_data": {
            "source": source, "reason": reason}})

    count = await rag.ingest(text, source)
    await cache.invalidate()
    return IngestResponse(source=source, chunks_indexed=count)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes (unauthenticated)."""
    settings = get_settings()
    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
        "rag": rag is not None,
    }
    return HealthResponse(
        status="healthy" if all(checks.values()) else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_admin)])
async def get_metrics():
    return MetricsResponse(**metrics.summary)


@app.get("/cache/stats", dependencies=[Depends(require_admin)])
async def cache_stats():
    return await cache.get_stats()