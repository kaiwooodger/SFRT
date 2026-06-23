#!/usr/bin/env python3
"""Dispatch clean public entry points to the preserved implementation layer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: legacy_dispatch.py <entry_key> [args...]")

    entry_key = sys.argv[1]
    script_dir = Path(__file__).resolve().parent
    legacy_root = script_dir / "internal_legacy"
    entrypoints = json.loads((legacy_root / "clean_entrypoints.json").read_text())

    if entry_key not in entrypoints:
        raise SystemExit(f"Unknown legacy entry key: {entry_key}")

    legacy_script = legacy_root / entrypoints[entry_key]
    completed = subprocess.run([sys.executable, str(legacy_script), *sys.argv[2:]])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
