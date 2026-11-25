import asyncio
import re
import time
import uuid
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config import get_settings
from monitoring import get_logger

from reranker import Reranker

logger = get_logger()


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def rrf(rank_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: merges ranked lists without normalizing scores."""
    scores: dict[tuple, float] = {}
    chunks: dict[tuple, RetrievedChunk] = {}
    for lst in rank_lists:
        for rank, c in enumerate(lst, 1):
            key = (c.source, c.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, c)
    return [chunks[key] for key in sorted(scores, key=scores.get, reverse=True)]


class RAGPipeline:
    """Chunk -> embed -> store (ingest) and embed -> search (retrieve)."""

    def __init__(self):
        s = get_settings()
        self.table = s.rag_table  
        self.top_k = s.rag_top_k
        self.min_score = s.rag_min_score  # similiarity threshold
        self.embed_dim = s.embedding_dim
        self.candidates = s.rag_candidates
        self.min_rerank_score = s.rag_min_rerank_score
        self.hybrid = s.rag_hybrid
        self._reranker = Reranker(s.reranker_model) if s.reranker_model else None

        self._embedder = OpenAIEmbeddings(
            model=s.embedding_model,
            api_key=s.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            chunk_size=64,
            timeout=30,
            max_retries=2,
        )
        # database_url must use the asyncpg driver:
        # postgresql+asyncpg://user:pass@host:5432/db
        self._engine = create_async_engine(s.database_url, pool_size=5)
        self._pg_engine = PGEngine.from_engine(self._engine)
        self._store: PGVectorStore | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )

    async def setup(self):
        """Create extension + table + store. Call once at startup."""
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            exists = await conn.scalar(
                text("SELECT to_regclass(:name) IS NOT NULL"),
                {"name": f"public.{self.table}"}
            )
        
        if not exists:
            await self._pg_engine.ainit_vectorstore_table(
                table_name=self.table,
                vector_size=self.embed_dim,
                metadata_columns=[Column("source", "TEXT", nullable=False)],
            )

        self._store = await PGVectorStore.create(
            engine=self._pg_engine,
            table_name=self.table,
            embedding_service=self._embedder,
            metadata_columns=["source"],
        )

        # Full-text index for the keyword leg (safe to run on every startup)
        async with self._engine.begin() as conn:
            await conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS "{self.table}_fts_idx" ON "{self.table}" '
                f"USING GIN (to_tsvector('english', content))"
            ))
        logger.info(f"pgvector table '{self.table}' ready (hybrid={self.hybrid})")

    @traceable(name="rag_ingest")
    async def ingest(self, text_: str, source: str) -> int:
        """Idempotent per source: re-ingesting replaces the old chunks."""
        text_ = _CONTROL_CHARS.sub("", text_)
        docs = self._splitter.create_documents(
            [text_], metadatas=[{"source": source}]
        )
        if not docs:
            return 0

        # Drop old chunks so edits don't leave stale ones
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.table} WHERE source = :source"),
                {"source": source},
            )

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")) for i in range(len(docs))]
        await self._store.aadd_documents(docs, ids=ids)
        return len(docs)

    async def _vector_search(self, query: str, n: int) -> list[RetrievedChunk]:
        results = await self._store.asimilarity_search_with_score(query, k=n)
        return [
            RetrievedChunk(text=doc.page_content,
                           source=doc.metadata["source"],
                           score=1.0 - distance)
            for doc, distance in results
        ]

    async def _keyword_search(self, query: str, n: int) -> list[RetrievedChunk]:
        """Postgres full-text search (not true BM25). Query words are OR-ed together."""
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
        if not words:
            return []
        tsq = " or ".join(words)  # websearch_to_tsquery treats 'or' as OR and never errors
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(
                f'SELECT source, content, ts_rank_cd(to_tsvector(\'english\', content), q) AS r '
                f'FROM "{self.table}", websearch_to_tsquery(\'english\', :q) q '
                f"WHERE to_tsvector('english', content) @@ q "
                f"ORDER BY r DESC LIMIT :n"
            ), {"q": tsq, "n": n})).all()
        return [RetrievedChunk(text=c, source=s, score=float(r)) for s, c, r in rows]

    @traceable(name="rag_retrieve")
    async def retrieve(self, query: str, k: int | None = None,
                       apply_threshold: bool = True,
                       rerank: bool = True) -> list[RetrievedChunk]:
        k = k or self.top_k
        use_rerank = rerank and self._reranker is not None
        n = max(k, self.candidates) if use_rerank else k

        if self.hybrid:
            vec, kw = await asyncio.gather(
                self._vector_search(query, n), self._keyword_search(query, n))
            chunks = rrf([vec, kw])[:n]
        else:
            chunks = await self._vector_search(query, n)

        if use_rerank:
            t = time.perf_counter()
            try:
                chunks = (await asyncio.wait_for(
                    self._reranker.rerank(query, chunks), timeout=8))[:k]
                threshold = self.min_rerank_score
            except asyncio.TimeoutError:
                logger.warning("rerank timed out; using first-stage ranking")
                chunks = chunks[:k]
                threshold = None
            logger.info("rerank done", extra={"extra_data": {
                "ms": round((time.perf_counter() - t) * 1000), "n": n}})
        else:
            chunks = chunks[:k]
            # a cosine threshold doesn't apply to keyword-only hits
            threshold = None if self.hybrid else self.min_score

        if apply_threshold and threshold is not None:
            chunks = [c for c in chunks if c.score >= threshold]
        return chunks
    
    async def close(self):
        await self._engine.dispose()