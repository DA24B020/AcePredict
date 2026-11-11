#!/usr/bin/env bash
# Local CI, run inside the pipeline-runner container (same image as production):
#   1. lint (ruff, PEP8)   2. unit tests + JUnit report   3. dvc repro (only changed stages)
#   4. dvc metrics diff vs the last commit   5. dvc dag   6. dvc push
#   scripts/ci.sh            full CI
#   scripts/ci.sh --fast     lint + tests only
source "$(dirname "$0")/common.sh"
$DC ps --status running --services | grep -qx pipeline-runner || die "Stack not running: scripts/start.sh"
say "1/6 ruff (PEP8 + imports)"
in_runner ruff check src backend frontend tests scripts
say "2/6 unit tests"
in_runner python -m pytest -q --junitxml=reports/junit.xml --cov=acepredict --cov=app \
  --cov-report=term-missing:skip-covered
in_runner python scripts/make_test_report.py reports/junit.xml
[ "${1:-}" = "--fast" ] && { say "fast CI passed"; exit 0; }
[ -f runtime/pipeline.lock ] && die "A pipeline run is in progress; try again when it finishes"
say "3/6 dvc repro"
in_runner dvc repro
say "4/6 dvc metrics diff (vs HEAD)"
in_runner dvc metrics diff HEAD || true
say "5/6 dvc dag"
in_runner dvc dag
say "6/6 dvc push"
in_runner dvc push
say "CI passed. Commit with scripts/version.sh"
