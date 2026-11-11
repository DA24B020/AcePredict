#!/usr/bin/env bash
# Queue one pipeline run (dvc repro) for the pipeline-runner service and follow its log.
# The run shows up in the UI's Pipeline Console, in Grafana and in MLflow.
#   scripts/run_pipeline.sh [reason]
source "$(dirname "$0")/common.sh"
reason="${1:-manual}"
$DC ps --status running --services | grep -qx pipeline-runner || die "Stack not running: scripts/start.sh"
[ -f runtime/pipeline.lock ] && die "A run is already in progress (runtime/pipeline.lock)"
run_id=$(date -u +%Y%m%dT%H%M%S)
mkdir -p runtime/triggers
printf '{"run_id": "%s", "reason": "%s", "queued_at": "%s"}' "$run_id" "$reason" "$(date -u +%FT%TZ)" \
  > "runtime/triggers/.${run_id}.tmp"
mv "runtime/triggers/.${run_id}.tmp" "runtime/triggers/${run_id}.json"
say "Queued run $run_id (reason: $reason). Following runtime/logs/run_${run_id}.log  (Ctrl-C stops following, not the run)"
log="runtime/logs/run_${run_id}.log"
for _ in $(seq 1 30); do [ -f "$log" ] && break; sleep 1; done
[ -f "$log" ] || die "Runner did not pick up the trigger; see: $DC logs pipeline-runner"
tail -n +1 -F "$log" &
tail_pid=$!
trap 'kill $tail_pid 2>/dev/null || true' EXIT
while [ -f runtime/pipeline.lock ]; do sleep 3; done
sleep 1; kill $tail_pid 2>/dev/null || true
status=$(python3 -c "import json;r=[x for x in json.load(open('runtime/pipeline_runs.json')) if x['run_id']=='$run_id'];print(r[-1]['status'] if r else 'unknown')")
if [ "$status" = "succeeded" ]; then
  say "Run $run_id succeeded"
  in_runner dvc metrics show || true
  say "Versioning: commit the new dvc.lock and push data to the DVC remote with scripts/version.sh"
else
  die "Run $run_id $status — open the Pipeline Console in the app for the failing stage"
fi
