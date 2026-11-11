#!/usr/bin/env bash
# Pick up newly published matches: ingest has no file deps on the remote CSVs, so force it
# and let DVC rerun everything downstream that changed.
source "$(dirname "$0")/common.sh"
in_runner dvc repro -f -s ingest
in_runner dvc repro
