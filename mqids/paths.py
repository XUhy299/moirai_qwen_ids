"""Project-relative paths for the independent MOIRAI-Qwen experiment."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Keep every runtime asset below PACKAGE_ROOT so this project can be copied to
# another machine or Linux host without depending on its former parent repo.
MODEL_ROOT = PACKAGE_ROOT / "models"
MOIRAI_MODEL_ROOT = MODEL_ROOT
QWEN_MODEL_ROOT = MODEL_ROOT

EXPORTED_DATA_ROOT = PACKAGE_ROOT / "data" / "wadi"
VAL_TEST_ROOT = EXPORTED_DATA_ROOT / "val_test"

WADI_TRAIN_X = EXPORTED_DATA_ROOT / "WADI-CLEAN_X_train.npy"
WADI_VAL_X = VAL_TEST_ROOT / "WADI-CLEAN_X_test_val.npy"
WADI_VAL_Y = VAL_TEST_ROOT / "WADI-CLEAN_Y_test_val.npy"
WADI_TEST_X = VAL_TEST_ROOT / "WADI-CLEAN_X_test_new.npy"
WADI_TEST_Y = VAL_TEST_ROOT / "WADI-CLEAN_Y_test_new.npy"
WADI_SENSOR_NAMES = EXPORTED_DATA_ROOT / "WADI-CLEAN_sensor_cols.txt"
WADI_SCALER = EXPORTED_DATA_ROOT / "WADI-CLEAN_scaler.pkl"

OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def package_relative_path(path: Path) -> str:
    """Return a portable, POSIX-style path for metadata written by this project."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def moirai_model_path(size: str) -> Path:
    size = size.strip().lower()
    if size not in {"small", "base", "large"}:
        raise ValueError(f"Unsupported MOIRAI size: {size!r}")
    return MOIRAI_MODEL_ROOT / f"moirai-1.1-R-{size}"


def qwen_model_path(subdir: str = "Qwen3-0.6B") -> Path:
    return QWEN_MODEL_ROOT / subdir


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required project files:\n" + "\n".join(missing))


def guard_development_paths(*paths: Path) -> None:
    """Fail closed if a training/development entry point requests formal test data."""
    forbidden = {WADI_TEST_X.resolve(), WADI_TEST_Y.resolve()}
    requested = {Path(path).resolve() for path in paths}
    overlap = sorted(str(path) for path in requested & forbidden)
    if overlap:
        raise RuntimeError("Development code requested locked WADI test data:\n" + "\n".join(overlap))
