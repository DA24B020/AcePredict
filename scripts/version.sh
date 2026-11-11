#!/usr/bin/env bash
# Version the current pipeline state: dvc push (data + models to the DVC remote),
# then a Git commit of dvc.lock / params / metrics. Reproduce any version later with
#   git checkout <commit> && dvc pull && dvc repro
source "$(dirname "$0")/common.sh"
msg="${1:-pipeline run $(date -u +%F\ %H:%M)}"
say "dvc push"
in_runner dvc push
say "git commit"
git add -A
git commit -m "$msg" || warn "nothing to commit"
git log --oneline | head -3
