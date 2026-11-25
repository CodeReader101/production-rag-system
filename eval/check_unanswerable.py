import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio, json, re, sys
from dotenv import load_dotenv; load_dotenv()
from sqlalchemy import text
from rag import RAGPipeline

async def main(path):
    rag = RAGPipeline(); await rag.setup()
    async with rag._engine.connect() as conn:
        rows = (await conn.execute(text(f'SELECT source, content FROM "{rag.table}"'))).all()
    for line in open(path):
        q = json.loads(line)
        kws = q.get("check_keywords", [])
        if not kws:
            print(f'\n{q["question"]}\n  (no check needed)')
            continue
        src = q.get("check_source")
        pool = [(s, c) for s, c in rows if src is None or s == src]
        hits = [(s, c) for s, c in pool if any(k in c.lower() for k in kws)]
        print(f'\n{q["question"]}\n  keyword hits in {src or "all"}: {len(hits)}')
        for s, c in hits[:3]:
            i = min(c.lower().find(k) for k in q["check_keywords"] if k in c.lower())
            print(f'  [{s}] ...{re.sub(chr(10), " ", c[max(0, i-80): i+120])}...')
    await rag.close()

asyncio.run(main(sys.argv[1]))