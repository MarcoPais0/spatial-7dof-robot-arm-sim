#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gui_environment import collect_runtime_issues, format_runtime_issues


def main() -> int:
    issues = collect_runtime_issues()
    print(format_runtime_issues(issues))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
