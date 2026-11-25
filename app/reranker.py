import asyncio
from dataclasses import replace

from sentence_transformers import CrossEncoder

class Reranker:
    """Cross-encoder reranker. Scores (query, chunk) pairs jointly."""

    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, chunks: list) -> list:
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        # predict() is blocking CPU work, so keep it off the event loop
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [replace(c, score=float(s)) for c, s in ranked]