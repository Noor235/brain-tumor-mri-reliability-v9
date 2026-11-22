#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f src/step22_full_test_reliability.py ]]; then
  echo "ERROR: Install package in the existing V9 project root (folder with src/)." >&2; exit 1
fi
if [[ ! -f src/research_v9.py ]]; then
  echo "ERROR: The original V9 src/research_v9.py was not found." >&2; exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m src.step22_full_test_reliability --layers final earlier "$@"
