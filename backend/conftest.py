"""Root conftest for backend tests. Ensures backend/ is on sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
