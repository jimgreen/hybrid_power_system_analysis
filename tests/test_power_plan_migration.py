import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POWER_PLAN = ROOT / "power_plan"
LEGACY_SITE = ROOT / ("web_" + "monitor")


def test_power_plan_replaces_legacy_site_directory():
    assert POWER_PLAN.exists()
    assert not LEGACY_SITE.exists()


def test_power_plan_server_resolves_planning_page():
    sys.path.insert(0, str(POWER_PLAN))
    try:
        server = importlib.import_module("server")
    finally:
        sys.path.remove(str(POWER_PLAN))

    assert server.WEB_ROOT == POWER_PLAN
    assert server.resolve_static_path("/planning.html") == POWER_PLAN / "planning.html"
