import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# eval/run_retrieval_eval.py
import asyncio, json
from collections import defaultdict
from dotenv import load_dotenv; load_dotenv()
from rag import RAGPipeline

K = 5

def norm(s: str) -> str:
    return " ".join(s.split()).lower()

async def main():
    rag = RAGPipeline()
    await rag.setup()
    rag.min_score = -1.0   # evaluate ranking, not the threshold
    rag.top_k = K

    rows = [json.loads(l) for l in open("eval/golden.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["expected_snippet"]]

    by_src = defaultdict(lambda: [0, 0, 0.0])    # hits, total, sum of reciprocal ranks
    by_type = defaultdict(lambda: [0, 0, 0.0])
    wrong_source = 0

    for row in rows:
        chunks = await rag.retrieve(row["question"])
        rank = next((i + 1 for i, c in enumerate(chunks)
                     if norm(row["expected_snippet"]) in norm(c.text)), None)
        for key, table in ((row["source"], by_src), (row["type"], by_type)):
            table[key][1] += 1
            if rank:
                table[key][0] += 1
                table[key][2] += 1 / rank
        if not rank:
            print("MISS:", row["question"])
            for c in chunks[:3]:
                print(f"    {c.score:.3f} [{c.source}] {c.text[:90]!r}")
        if chunks and chunks[0].source != row["source"]:
            wrong_source += 1

    def report(title, table):
        print(f"\n{title}")
        for k, (h, n, rr) in sorted(table.items()):
            print(f"  {k:34s} hit@{K} {h}/{n} = {h/n:.2f}   MRR {rr/n:.2f}")

    report("By source", by_src)
    report("By type", by_type)
    H = sum(v[0] for v in by_src.values()); N = sum(v[1] for v in by_src.values())
    RR = sum(v[2] for v in by_src.values())
    print(f"\nOVERALL hit@{K} {H}/{N} = {H/N:.2f}   MRR {RR/N:.2f}")
    print(f"Top-1 chunk from the wrong document: {wrong_source}/{N}")
    await rag.close()

asyncio.run(main())