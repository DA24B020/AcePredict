#!/usr/bin/env bash
# One-time setup on the VM: checks tools, frees ports, creates .env, builds the images,
# and initialises Git + Git LFS + DVC (with a local DVC remote in ~/dvc-store).
source "$(dirname "$0")/common.sh"

say "Checking tools"
command -v docker >/dev/null || die "Docker is not installed (sudo apt install docker.io docker-compose-v2)"
docker info >/dev/null 2>&1 || die "Docker is not usable by $(whoami). Run: sudo usermod -aG docker $USER && newgrp docker"
command -v git >/dev/null || die "git missing: sudo apt install git"
if ! git lfs version >/dev/null 2>&1; then
  say "Installing git-lfs"; sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi
command -v curl >/dev/null || sudo apt-get install -y -qq curl

free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -lt 6 ] && warn "Only ${free_gb} GB free; the images + data need about 5-6 GB."

say "Freeing ports used by the stack (old Grafana from assignment 4, etc.)"
for c in $(docker ps -q --filter "publish=3000" --filter "publish=9090" --filter "publish=9093"); do
  name=$(docker inspect -f '{{.Name}}' "$c" | tr -d /)
  case "$name" in acepredict*) ;; *) warn "stopping container $name"; docker stop "$c" >/dev/null;; esac
done
docker ps --format '{{.Names}}' | grep -qx grafana && { warn "stopping old container grafana"; docker stop grafana >/dev/null; }
for svc in grafana-server prometheus alertmanager prometheus-node-exporter; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then warn "stopping system service $svc"; sudo systemctl stop "$svc"; fi
done
# Prometheus / Alertmanager / node_exporter started by hand from ~/monitoring (not in Docker)
for port in 3000 5000 5001 8000 8501 9090 9091 9093; do
  line=$(ss -ltnpH "sport = :$port" 2>/dev/null | head -1 || true)
  if [ -n "$line" ] && ! echo "$line" | grep -q docker-proxy; then
    warn "port $port is busy: $line"
    warn "  stop that program (e.g. pkill -f prometheus) before scripts/start.sh"
  fi
done

say "Creating .env"
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env
  sed -i "s#^DVC_REMOTE_DIR=.*#DVC_REMOTE_DIR=$HOME/dvc-store#" .env
fi
mkdir -p runtime/logs runtime/triggers "$HOME/dvc-store"
chmod +x scripts/*.sh docker/entrypoint.sh

say "Git + Git LFS"
[ -d .git ] || git init -q -b main
git config user.email >/dev/null || git config user.email "student@acepredict.local"
git config user.name  >/dev/null || git config user.name  "AcePredict Student"
git lfs install --local >/dev/null
git config core.hooksPath >/dev/null || true

say "Building Docker images (first time ~10 min, ~3.5 GB)"
$DC build

say "DVC init + local remote"
if [ ! -d .dvc ]; then
  $DC run --rm --no-deps -T pipeline-runner dvc init -q
fi
$DC run --rm --no-deps -T pipeline-runner sh -c '
  dvc remote add -d -f localstore /dvc-store &&
  dvc config cache.type "hardlink,symlink" &&
  dvc config core.autostage true'

say "First commit"
git add -A
git commit -qm "AcePredict: initial project" || true
git log --oneline | head -3

say "Setup done. Next: scripts/start.sh, then scripts/run_pipeline.sh"
