import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# Set before anything imports config. Real env vars beat .env, so tests never
# use your real keys and never send traces to LangSmith.
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://u:p@localhost:5432/db"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"