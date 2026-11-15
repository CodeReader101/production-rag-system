# eval/check_golden.py  (run with PYTHONPATH=app)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio, json
from dotenv import load_dotenv; load_dotenv()
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from config import get_settings

def norm(s): return " ".join(s.split()).lower()

async def main():
    s = get_settings()
    eng = create_async_engine(s.database_url)
    async with eng.connect() as conn:
        rows = (await conn.execute(text(f"SELECT source, content FROM {s.rag_table}"))).all()
    chunks = [(src, norm(c)) for src, c in rows]
    for line in open("eval/golden.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r.get("expected_snippet"):
            continue
        hits = [src for src, c in chunks if norm(r["expected_snippet"]) in c]
        flag = "OK " if hits and r.get("source") in hits else "BAD"
        print(f"{flag} {r['question'][:60]!r} -> snippet in {sorted(set(hits)) or 'NO CHUNK'}")
    await eng.dispose()

asyncio.run(main())