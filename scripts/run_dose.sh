#!/usr/bin/env bash
# BF16 dose finding on Qwen and Gemma, validation split only:
#   environment check -> offline tests -> Gemma access check
#   -> for each model: vectors, 50-condition sweep, report, verdict -> results ZIP
# A FAIL verdict is a result, not an error, so the run still finishes and packages.
# Start it with `bash run_experiment.sh dose`. Running that again resumes.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RESUME="bash run_experiment.sh dose"
source scripts/campaign_lib.sh

status "STARTED"
stage dose-environment python scripts/check_gpu.py --output logs/dose-gpu-environment.json
python -m pip freeze > logs/dose-packages.txt
once dose-tests python -m pytest -q
stage dose-access python scripts/check_model_access.py google/gemma-2-9b-it

for model in qwen gemma; do
  config=configs/dose_$model
  root=artifacts/dose-$model
  stage "dose-$model-plan" steerquant plan --config "$config.yaml"
  lock_once "$config"
  extract_once "$config" "$root" bf16 "dose-$model"
  once "dose-$model-validation" steerquant evaluate --config "$config.locked.json" --precision bf16 --split validation --resume
  summarize_once "$config" validation "reports/dose-$model" "dose-$model-report"
  stage "dose-$model-verdict" steerquant dose-check --config "$config.locked.json" --output "reports/dose-$model/verdict.json"
done

ZIP="dose-results-$(date -u +%Y%m%dT%H%M%SZ).zip"
stage dose-package python scripts/package_results.py --output "$ZIP" --only dose \
  --artifact-root artifacts/dose-qwen --artifact-root artifacts/dose-gemma
status "DONE. Verdicts: reports/dose-qwen/verdict.md, reports/dose-gemma/verdict.md. Results archive: $ZIP"
