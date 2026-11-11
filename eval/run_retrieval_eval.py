# eval/run_retrieval_eval.py
import asyncio, json
from rag import RAGPipeline

K = 5


async def main():
    rag = RAGPipeline()
    await rag.setup()
    rag.min_score = -1.0   # evaluate ranking quality, not the threshold
    rag.top_k = K

    def norm(s: str) -> str:
        return " ".join(s.split()).lower()

    rows = [json.loads(l) for l in open("eval/golden.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["expected_snippet"]]   # skip unanswerable here
    hits, rr = 0, 0.0
    for row in rows:
        chunks = await rag.retrieve(row["question"])
        rank = next(
            (i + 1 for i, c in enumerate(chunks)
            if norm(row["expected_snippet"]) in norm(c.text)),
            None,
        )
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            print("MISS:", row["question"])
            for c in chunks[:3]:
                print(f"    {c.score:.3f} [{c.source}] {c.text[:90]!r}")
                
    n = len(rows)
    print(f"hit@{K}: {hits/n:.2f}   MRR: {rr/n:.2f}   (n={n})")
    await rag.close()

asyncio.run(main())