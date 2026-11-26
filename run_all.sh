#!/usr/bin/env bash
# Reproduces every result, log and figure. Edit configs/config.json (roll_number) first.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
mkdir -p ../logs
$PY -c "from common import *; c=load_config(); print('seed', c['seed']); get_split(c)"
$PY q1.py --part a 2>&1 | tee ../logs/q1a.log
$PY q1_lr_edge.py 2>&1 | tee ../logs/q1a_lr_edge.log
$PY q1.py --part b 2>&1 | tee ../logs/q1b.log
$PY q2.py          2>&1 | tee ../logs/q2.log
$PY q3.py          2>&1 | tee ../logs/q3.log
$PY q4.py          2>&1 | tee ../logs/q4.log
$PY figures.py     2>&1 | tee ../logs/figures.log
$PY tables.py
cd ../report && tectonic report.tex
