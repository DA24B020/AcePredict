#!/usr/bin/env bash
# Switch the data source between the real Sackmann CSVs (github) and the small synthetic
# sample (sample, no internet needed, whole pipeline in ~3 min). DVC reruns from ingest.
#   scripts/use_sample_data.sh sample | github
source "$(dirname "$0")/common.sh"
src="${1:-sample}"
case "$src" in sample|github) ;; *) die "usage: $0 sample|github";; esac
sed -i "s/^\(  source: \)[a-z]*/\1$src/" params.yaml
grep -n "^  source:" params.yaml
say "Now run scripts/run_pipeline.sh"
