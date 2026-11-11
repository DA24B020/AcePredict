# Shared helpers, sourced by the other scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
env_get() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2-; }
# Run a command inside the pipeline-runner container (has DVC, Spark, Ray, pytest).
in_runner() { $DC exec -T pipeline-runner "$@"; }
wait_http() {  # wait_http URL SECONDS
  local url=$1 t=${2:-120} i=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    i=$((i + 2)); [ "$i" -ge "$t" ] && return 1; sleep 2
  done
}
