#!/usr/bin/env bash
# Start the whole stack (builds images if they are missing) and print the URLs.
source "$(dirname "$0")/common.sh"
[ -f .env ] || die "Run scripts/setup.sh first"
mkdir -p runtime/logs runtime/triggers
if ! docker image inspect acepredict-ml:latest >/dev/null 2>&1 || [ "${1:-}" = "--build" ]; then
  say "Building images"; $DC build
fi
say "Starting services"
$DC up -d
say "Waiting for the backend"
wait_http http://localhost:8000/health 180 || { $DC ps; die "backend did not start; see: $DC logs backend"; }
ready=$(curl -s http://localhost:8000/ready || true)
$DC ps --format 'table {{.Name}}\t{{.Status}}'
cat <<TXT

  App (Streamlit)        http://localhost:8501
  API docs (FastAPI)     http://localhost:8000/docs
  MLflow                 http://localhost:5000
  Grafana                http://localhost:3000   (admin / $(env_get GRAFANA_ADMIN_PASSWORD))
  Prometheus             http://localhost:9090
  Alertmanager           http://localhost:9093
  Pushgateway            http://localhost:9091
  Ray dashboard          http://localhost:8265   (only while the tune_elo stage runs)
  Spark UI               http://localhost:4040   (only while the transform stage runs)

  /ready -> $ready
TXT
echo "$ready" | grep -q '"ready":true' || warn "No model/snapshot yet: run scripts/run_pipeline.sh"
