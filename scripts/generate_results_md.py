"""Backfill human-readable results.md files for existing experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mqids.training import write_run_markdown


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="*", type=Path)
    args = parser.parse_args()
    roots = args.outputs or sorted(path for path in (PACKAGE_ROOT / "outputs").iterdir() if path.is_dir())
    generated = 0
    skipped = 0
    for output_dir in roots:
        required = [
            output_dir / "config.json",
            output_dir / "protocol.json",
            output_dir / "environment.json",
        ]
        if not all(path.exists() for path in required):
            skipped += 1
            continue
        history_path = output_dir / "history.json"
        history = load_json(history_path).get("epochs", []) if history_path.exists() else []
        status = "completed" if history else "prepare_only"
        write_run_markdown(
            output_dir,
            config=load_json(required[0]),
            protocol=load_json(required[1]),
            environment=load_json(required[2]),
            history=history,
            status=status,
        )
        generated += 1
    print(f"generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
