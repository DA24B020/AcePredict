#!/usr/bin/env bash
# Replay the held-out season against the running API (feeds live accuracy + drift panels).
#   scripts/replay.sh                       normal replay
#   scripts/replay.sh --inject-drift        input drift -> FeatureDrift alert -> auto retrain
#   scripts/replay.sh --flip-outcomes 0.7   wrong results -> LiveLogLossHigh alert
source "$(dirname "$0")/common.sh"
in_runner python scripts/replay_season.py --base-url http://backend:8000 "$@"
