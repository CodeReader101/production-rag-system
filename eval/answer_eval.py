import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio, json, os, re, sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from config import get_settings
from rag import RAGPipeline
from agent import ProductionAgent

RERANK = os.environ.get("RERANK", "1") == "1"
JUDGE_MODEL = os.environ["JUDGE_MODEL"]  # use a different model than primary_model if you can

ANSWERABLE_PROMPT = """You are grading a RAG system's answer.

Question: {q}
Reference answer: {ref}
Retrieved context:
{ctx}

System answer: {ans}

Return ONLY JSON: {{"correct": true/false, "grounded": true/false, "refused": true/false, "reason": "one short sentence"}}
- correct: the system answer agrees with the reference answer on the key fact(s). Extra correct detail is fine.
- grounded: every factual claim in the system answer is supported by the retrieved context.
- refused: the system says it does not have the information."""

UNANSWERABLE_PROMPT = """You are grading a RAG system. The documents do NOT contain the answer to this question.

Question: {q}
Retrieved context:
{ctx}

System answer: {ans}

Return ONLY JSON: {{"refused": true/false, "fabricated": true/false, "reason": "one short sentence"}}
- refused: the system says the information is not available. It may add what the documents DO say (for example "the paper reports cost in FLOPs, not dollars"), which is still a refusal.
- fabricated: the system gives a specific answer that is not supported by the context, or answers from outside knowledge."""


async def judge(client, prompt: str) -> dict:
    for attempt in range(3):
        try:
            r = await client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = r.choices[0].message.content
            return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception as e:
            await asyncio.sleep(2 * (attempt + 1))
    return {"judge_error": True}


async def main(path: str, tag: str):
    s = get_settings()
    rag = RAGPipeline()
    await rag.setup()
    if not RERANK:
        rag._reranker = None
    agent = ProductionAgent(rag)
    client = AsyncOpenAI(api_key=s.openrouter_api_key.get_secret_value(),
                         base_url="https://openrouter.ai/api/v1", timeout=60)

    golden = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = []
    for i, g in enumerate(golden, 1):
        unans = g["source"] is None
        res = await agent.ainvoke(g["question"])
        failed = res["model_used"] == "error_handler" or res.get("error")
        verdict = {"skipped": "agent error"} if failed else await judge(
            client,
            (UNANSWERABLE_PROMPT if unans else ANSWERABLE_PROMPT).format(
                q=g["question"], ref=g["expected_answer"],
                ctx=res["context"] or "(none)", ans=res["response"]),
        )
        rows.append({"question": g["question"], "unanswerable": unans, "type": g["type"],
                     "answer": res["response"], "model_used": res["model_used"], **verdict})
        print(f"[{i}/{len(golden)}] {g['question'][:70]}")
        await asyncio.sleep(1)  # be gentle with free-tier rate limits

    ok = [r for r in rows if "judge_error" not in r and "skipped" not in r]
    ans = [r for r in ok if not r["unanswerable"]]
    una = [r for r in ok if r["unanswerable"]]
    pct = lambda n, d: f"{n}/{d} = {n / d:.0%}" if d else "n/a"

    print(f"\n== ANSWER-LEVEL RESULTS ({tag}, rerank={RERANK}) ==")
    print(f"graded {len(ok)} of {len(rows)} (errors/skipped: {len(rows) - len(ok)})")
    print("Answerable:")
    print("  correct       ", pct(sum(r.get("correct", False) for r in ans), len(ans)))
    print("  grounded      ", pct(sum(r.get("grounded", False) for r in ans), len(ans)))
    print("  false refusal ", pct(sum(r.get("refused", False) for r in ans), len(ans)))
    print("Unanswerable:")
    print("  correct refusal", pct(sum(r.get("refused", False) for r in una), len(una)))
    print("  fabricated     ", pct(sum(r.get("fabricated", False) for r in una), len(una)))

    print("\n== FAILURES ==")
    for r in ok:
        bad = (r["unanswerable"] and not r.get("refused")) or \
              (not r["unanswerable"] and (not r.get("correct") or r.get("refused")))
        if bad:
            print(f"- [{r['type']}] {r['question']}\n    answer: {r['answer'][:200]!r}\n    judge: {r.get('reason')}")

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/answers_{tag}.json").write_text(json.dumps(rows, indent=2))
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.jsonl",
                     sys.argv[2] if len(sys.argv) > 2 else "answers"))