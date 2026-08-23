"""Put the repo root on sys.path so root-level tests can import `shared`.

This dir intentionally has no __init__.py (it is not the `tests` package), so
it never shadows the role-scoped `tests` packages.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
