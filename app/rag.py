import uuid
from dataclasses import dataclass
import re

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
        logger.info(f"pgvector table '{self.table}' ready")

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


    @traceable(name="rag_retrieve")
    async def retrieve(self, query: str, k: int | None = None,
                       apply_threshold: bool = True,
                       rerank: bool = True) -> list[RetrievedChunk]:
        k = k or self.top_k
        use_rerank = rerank and self._reranker is not None
        n = max(k, self.candidates) if use_rerank else k

        results = await self._store.asimilarity_search_with_score(query, k=n)
        chunks = [
            RetrievedChunk(text=doc.page_content,
                           source=doc.metadata["source"],
                           score=1.0 - distance)
            for doc, distance in results
        ]

        if use_rerank:
            chunks = (await self._reranker.rerank(query, chunks))[:k]
            threshold = self.min_rerank_score
        else:
            threshold = self.min_score

        if apply_threshold:
            chunks = [c for c in chunks if c.score >= threshold]
        return chunks

    async def close(self):
        await self._engine.dispose()