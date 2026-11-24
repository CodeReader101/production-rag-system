import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import os
RERANK = os.environ.get("RERANK", "1") == "1"

import asyncio, json, re, statistics, sys, time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from rag import RAGPipeline

KS = (1, 3, 5, 10, 20, 30)


def squash(s: str) -> str:
    """Lowercase and drop all whitespace so PDF-extraction spacing can't cause false misses."""
    return re.sub(r"\s+", "", s.lower())


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {f"hit@{k}": sum(1 for r in rows if r["chunk_rank"] and r["chunk_rank"] <= k) / n
           for k in KS}
    out["mrr"] = sum(1 / r["chunk_rank"] for r in rows if r["chunk_rank"]) / n
    out["source@1"] = sum(1 for r in rows if r["source_rank"] == 1) / n
    out["n"] = n
    return out


def print_table(title: str, groups: dict[str, dict]):
    print(f"\n== {title} ==")
    print(f"{'group':34}{'n':>3}  " + "  ".join(f"hit@{k}" for k in KS) + "    mrr  src@1")
    for name, m in groups.items():
        print(f"{name:34}{m['n']:>3}  "
              + "  ".join(f"{m[f'hit@{k}']:5.2f}" for k in KS)
              + f"  {m['mrr']:5.2f}  {m['source@1']:5.2f}")


async def audit(rag: RAGPipeline, golden: list[dict]) -> list[dict]:
    """Find dataset/chunking problems before blaming retrieval."""
    async with rag._engine.connect() as conn:
        # 'content' is the default content column in PGVectorStore; verify for your version
        rows = (await conn.execute(text(f'SELECT source, content FROM "{rag.table}"'))).all()
    by_source = defaultdict(list)
    for src, content in rows:
        by_source[src].append(squash(content))

    problems = []
    for g in golden:
        chunks = by_source.get(g["source"])
        if not chunks:
            problems.append({**g, "issue": "source not ingested (name mismatch?)"})
        elif not any(squash(g["expected_snippet"]) in c for c in chunks):
            problems.append({**g, "issue": "snippet not in any single chunk "
                                           "(typo, extraction artifact, or split across chunks)"})
    return problems


async def main(path: str, tag: str):
    rag = RAGPipeline()
    await rag.setup()
    golden = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    problems = await audit(rag, golden)
    if problems:
        print(f"\n!! {len(problems)} golden items cannot be matched in the index:")
        for p in problems:
            print(f"  - [{p['source']}] {p['question']}\n      {p['issue']}: {p['expected_snippet']!r}")

    rows = []
    for g in golden:
        t0 = time.perf_counter()
        chunks = await rag.retrieve(g["question"], k=max(KS), apply_threshold=False, rerank=RERANK)
        latency_ms = (time.perf_counter() - t0) * 1000
        snip = squash(g["expected_snippet"])

        chunk_rank = next((i for i, c in enumerate(chunks, 1)
                           if c.source == g["source"] and snip in squash(c.text)), None)
        source_rank = next((i for i, c in enumerate(chunks, 1) if c.source == g["source"]), None)
        rows.append({
            **g,
            "chunk_rank": chunk_rank,
            "source_rank": source_rank,
            "hit_score": chunks[chunk_rank - 1].score if chunk_rank else None,
            "top1_score": chunks[0].score if chunks else None,
            "top1_source": chunks[0].source if chunks else None,
            "latency_ms": latency_ms,
        })

    by_type, by_source = defaultdict(list), defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
        by_source[r["source"]].append(r)
    print_table("OVERALL", {"all": summarize(rows)})
    print_table("BY QUESTION TYPE", {k: summarize(v) for k, v in by_type.items()})
    print_table("BY SOURCE", {k: summarize(v) for k, v in by_source.items()})

    # Threshold guidance: scores of the correct chunk when found
    hit_scores = sorted(r["hit_score"] for r in rows if r["hit_score"] is not None)
    if hit_scores:
        print(f"\nCorrect-chunk scores: min={hit_scores[0]:.2f} "
              f"median={statistics.median(hit_scores):.2f} max={hit_scores[-1]:.2f}")
        print(f"Current rag_min_score={rag.min_score}; it should sit below almost all of these.")

    lat = sorted(r["latency_ms"] for r in rows)
    print(f"Retrieval latency: p50={statistics.median(lat):.0f}ms "
          f"p95={lat[int(len(lat) * 0.95) - 1]:.0f}ms")

    misses = [r for r in rows if not r["chunk_rank"] or r["chunk_rank"] > 5]
    print(f"\n== MISSES (not in top 5): {len(misses)} ==")
    for r in misses:
        print(f"- [{r['type']}/{r['source']}] {r['question']}\n"
              f"    expected: {r['expected_snippet']!r} | rank={r['chunk_rank']} "
              f"| top1={r['top1_source']} ({r['top1_score']:.2f})")

    Path("eval_results").mkdir(exist_ok=True)
    out = {"tag": tag, "summary": summarize(rows), "rows": rows}
    Path(f"eval_results/{tag}.json").write_text(json.dumps(out, indent=2))
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "golden.jsonl",
                     sys.argv[2] if len(sys.argv) > 2 else "baseline"))