import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
PKG_DIR = SRC_DIR / "hybrid_power_system_analysis"
for path in (SRC_DIR, PKG_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from hybrid_power_system_analysis.drawcore.station_svg import main


if __name__ == "__main__":
    raise SystemExit(main())
