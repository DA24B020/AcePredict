#!/usr/bin/env bash
# Follow container logs:  scripts/logs.sh [service...]   e.g. scripts/logs.sh backend model-server
source "$(dirname "$0")/common.sh"
$DC logs -f --tail 100 "$@"
