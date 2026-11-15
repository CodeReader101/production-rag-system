# eval/find_chunks.py   usage: PYTHONUTF8=1 PYTHONPATH=app python eval/find_chunks.py "7.2 points"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio, sys
from dotenv import load_dotenv; load_dotenv()
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from config import get_settings

async def main(phrase):
    s = get_settings()
    eng = create_async_engine(s.database_url)
    async with eng.connect() as conn:
        rows = (await conn.execute(
            text(f"SELECT source, content FROM {s.rag_table} "
                 "WHERE regexp_replace(content, '\\s+', ' ', 'g') ILIKE :p ORDER BY source"),
            {"p": f"%{phrase}%"})).all()
    print(f"{len(rows)} chunk(s) contain {phrase!r}")
    for src, c in rows:
        i = " ".join(c.split()).lower().find(phrase.lower())
        print(f"  [{src}] ...{' '.join(c.split())[max(0, i-60): i+len(phrase)+60]}...")
    await eng.dispose()

asyncio.run(main(sys.argv[1]))