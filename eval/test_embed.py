# eval/test_embed.py  (run with PYTHONPATH=app)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio
from dotenv import load_dotenv; load_dotenv()
from config import get_settings
from langchain_openai import OpenAIEmbeddings

async def main():
    s = get_settings()
    e = OpenAIEmbeddings(model=s.embedding_model, api_key=s.openrouter_api_key,
                         base_url="https://openrouter.ai/api/v1",
                         check_embedding_ctx_length=False, max_retries=0)
    v = await e.aembed_query("hello world")
    print("ok, dims =", len(v))

asyncio.run(main())