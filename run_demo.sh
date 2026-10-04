#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-demo-lock.txt
mkdir -p validation
{
  echo 'Synthetic-only verification; no original reviews or external model calls.'
  .venv/bin/python --version
  .venv/bin/python -m pytest -q
  .venv/bin/python demo.py
} 2>&1 | tee validation/latest.txt
