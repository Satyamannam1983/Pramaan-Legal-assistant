"""pytest configuration — sets up test environment before any app imports."""
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Must happen BEFORE any app imports so Settings() picks up these values
os.environ["EMBEDDING_PROVIDER"] = "stub"
os.environ["EMBEDDING_DIMENSION"] = "64"
os.environ["EMBEDDING_BATCH_SIZE"] = "8"

# Never let a developer's DATABASE_URL (which may target real data) implicitly
# become the test database. MongoDB-dependent tests opt in via TEST_DATABASE_URL.
test_database_url = os.getenv("TEST_DATABASE_URL")
if test_database_url:
    database_name = urlsplit(test_database_url).path.strip("/").split("/", 1)[0]
    if "test" not in database_name.lower():
        raise RuntimeError(
            "TEST_DATABASE_URL must point to a database whose name contains 'test'."
        )
    os.environ["DATABASE_URL"] = test_database_url
else:
    os.environ["DATABASE_URL"] = "mongodb://127.0.0.1:59999/test_nonexistent"

# Ensure backend package is importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
