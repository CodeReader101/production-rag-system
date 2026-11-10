from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable


from config import get_settings
from models import (
    ChatRequest, ChatResponse, HealthResponse, MetricsResponse, ErrorResponse
)

from security import SecurityPipeline
from cache import RedisResponseCache
from monitoring import get_logger, MetricsCollector, RequestTimer
from agent import ProductionAgent

load_dotenv()

security: SecurityPipeline = None
cache: RedisResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    """

    global security, cache, metrics, agent

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
    agent = ProductionAgent()

    logger.info("All components initialized. Ready to serve requests")

    yield

    logger.info("Shutting down...", extra={
        "extra_data": metrics.summary
    })
    await cache.close()


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
        cached_response = await cache.get(cleaned_message)
        if cached_response is not None:
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
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )


        # --- Step 3: Invoke LangGrpah Agent ---

        try:
            result = await agent.ainvoke(cleaned_message)
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
        if model_used != "error_handler":
            await cache.set(cleaned_message, validated_response)

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
         thread_id=body.thread_id,
         model_used=model_used,
         cached=False,
         processing_time_ms=round(timer.elapsed_ms, 2)
    )




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