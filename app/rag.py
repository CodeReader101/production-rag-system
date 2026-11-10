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

logger = get_logger()


@dataclass
class RetrievedChunk:
    text: set
    source: str
    score: float

class RAGPipeline:
    def __init__(self):
        settings = get_settings()
        self.table = settings.rag_table
        self.top_k = settings.rag_top_k
        self.min_score = settings.rag_min_score
        self.embed_dim = settings.embedding_dim

        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            chunk_size=64,
            timeout=30,
            max_retries=2,
        )

        self._engine = create_async_engine(settings.database_url, pool_size=5)
        self._pg_engine = PGEngine.from_engine(self._engine)
        self._store: PGVectorStore | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )

    async def setup(self):
        """Create extension + table + store. Call once at startup"""
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        await self._pg_engine.ainit_vectorstore_table(
            table_name=self.table,
            vector_size=self.embed_dim,
            metadata_column=[Column("source", "TEXT", nullable=False)],
        )
        self._store = await PGVectorStore.create(
            engine=self._pg_engine,
            table_name=self.table,
            embedding_service=self._embedder,
            metadata_column=["source"],
        )
        logger.info(f"pgvector table '{self.table} ready")

    @traceable(name="rag_ingest")
    async def ingest(self, text_: str, source: str) -> int:
        """Idempotent per source: re-ingesting replaces the old chunks."""
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

        id = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{1}")) for i in range(len(docs))]
        await self._store.aadd_documents(docs, ids=ids)
        return len(docs)

    @traceable(name="rag_retrieve")
    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        results = await self._store.asimilarity_search_with_score(query, k=self.top_k)
        chunks = []
        for doc, distance in results:
            score = 1.0 - distance 
            if score >= self.min_score:
                chunks.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        source=doc.metadat["source"],
                        score=score,
                    )
                )
        return chunks

    async def close(self):
        await self._engine.dispose()