import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR / "lfcore", ROOT_DIR / "model", ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_flow import *  # noqa: F401,F403
