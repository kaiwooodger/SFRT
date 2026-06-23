#!/usr/bin/env python3
\"\"\"Public entry point for generating the 10-case synthetic cohort.\"\"\"

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    dispatcher = Path(__file__).resolve().parent / "legacy_dispatch.py"
    completed = subprocess.run([sys.executable, str(dispatcher), "generate_synthetic_cohort", *sys.argv[1:]])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
