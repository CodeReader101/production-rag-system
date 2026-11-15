# eval/find_bad_chunk.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import asyncio, sys, re
from dotenv import load_dotenv; load_dotenv()
from config import get_settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

async def main(path):
    s = get_settings()
    e = OpenAIEmbeddings(model=s.embedding_model, api_key=s.openrouter_api_key,
                         base_url="https://openrouter.ai/api/v1",
                         check_embedding_ctx_length=False, max_retries=1)
    text = CTRL.sub("", "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages))
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
    for i, c in enumerate(chunks):
        try:
            await e.aembed_query(c)
        except Exception as ex:
            print(f"chunk {i} failed: {ex}\n{c!r}")
            return
    print(f"all {len(chunks)} chunks embedded fine")

asyncio.run(main(sys.argv[1]))