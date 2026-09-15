"""pytest configuration — sets up test environment before any app imports."""
import os
import sys
from pathlib import Path

# Must happen BEFORE any app imports so Settings() picks up these values
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["EMBEDDING_DIMENSION"] = "64"
os.environ["EMBEDDING_BATCH_SIZE"] = "8"

# Use a MongoDB URL that refuses connection immediately (port 59999 is not listening)
# so the async client fails fast instead of timing out
if not os.getenv("TEST_DATABASE_URL"):
    os.environ["DATABASE_URL"] = "mongodb://127.0.0.1:59999/test_nonexistent"

# Ensure backend package is importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
