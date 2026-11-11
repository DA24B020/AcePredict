#!/usr/bin/env bash
# Stop the stack. Data volumes (MLflow runs, Prometheus, Grafana) are kept.
# Use --wipe to also delete those volumes (fresh MLflow registry).
source "$(dirname "$0")/common.sh"
if [ "${1:-}" = "--wipe" ]; then $DC down -v; else $DC down; fi
