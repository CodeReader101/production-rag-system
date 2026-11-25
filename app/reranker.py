import asyncio
from dataclasses import replace


class Reranker:
    """Cross-encoder reranker. Scores (query, chunk) pairs jointly."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder   # moved out of the module top so it only import when reranker is needed not unnecssarly and increase the load time
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, chunks: list) -> list:
        if not chunks:
            return chunks
        pairs = [(query, c.text) for c in chunks]
        # predict() is blocking CPU work, so keep it off the event loop
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [replace(c, score=float(s)) for c, s in ranked]