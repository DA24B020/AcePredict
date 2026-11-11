#!/usr/bin/env bash
# Throughput / latency test of POST /predict (writes reports/load_test.json).
source "$(dirname "$0")/common.sh"
in_runner python scripts/load_test.py --base-url http://backend:8000 "$@"
