#!/usr/bin/env python3
\"\"\"Public entry point for bounded biology-parameter robustness analysis.\"\"\"

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    dispatcher = Path(__file__).resolve().parent / "legacy_dispatch.py"
    completed = subprocess.run([sys.executable, str(dispatcher), "run_biology_parameter_robustness", *sys.argv[1:]])
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
