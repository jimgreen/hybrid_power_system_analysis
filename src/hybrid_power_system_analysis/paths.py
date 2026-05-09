"""Project path helpers for the repository data layout."""

from pathlib import Path


def find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DATA_DIR = DATA_DIR / "model"
MEAS_DATA_DIR = DATA_DIR / "meas"
LF_PARAMETER_DIR = DATA_DIR / "lf"
SE_PARAMETER_DIR = DATA_DIR / "se"


def model_file(case_type: str, name: str) -> Path:
    return MODEL_DATA_DIR / case_type / name


def measurement_file(case_type: str, name: str) -> Path:
    return MEAS_DATA_DIR / case_type / name


def resolve_project_file(path: str | Path) -> Path:
    """Resolve old repository-relative data paths to the current data layout."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    text = candidate.as_posix()
    suffix = candidate.suffix.lower()
    if suffix == ".meas":
        rewrites = (
            ("data/ac/", "data/meas/ac/"),
            ("data/dc/", "data/meas/dc/"),
            ("data/hybrid/", "data/meas/hybrid/"),
            ("data/model/ac/", "data/meas/ac/"),
            ("data/model/dc/", "data/meas/dc/"),
            ("data/model/hybrid/", "data/meas/hybrid/"),
        )
    else:
        rewrites = ()
    for old, new in rewrites:
        if old in text:
            rewritten = Path(text.replace(old, new))
            if rewritten.exists():
                return rewritten
            if not rewritten.is_absolute():
                absolute = PROJECT_ROOT / rewritten
                if absolute.exists():
                    return absolute

    rewrites = (
        ("data/ac/", "data/model/ac/"),
        ("data/dc/", "data/model/dc/"),
        ("data/hybrid/", "data/model/hybrid/"),
        ("data/hybrid/", "data/ca/"),
        ("lf.para", "data/lf/lf.para"),
        ("se.para", "data/se/se.para"),
    )
    for old, new in rewrites:
        if old in text:
            rewritten = Path(text.replace(old, new))
            if rewritten.exists():
                return rewritten
            if not rewritten.is_absolute():
                absolute = PROJECT_ROOT / rewritten
                if absolute.exists():
                    return absolute

    if not candidate.is_absolute():
        absolute = PROJECT_ROOT / candidate
        if absolute.exists():
            return absolute
    return candidate
