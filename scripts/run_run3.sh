#!/usr/bin/env bash
# Run 3 (docs/run3.md), BF16 and validation only. The test split is never touched.
#   environment -> offline tests -> config check -> Gemma access
#   -> 1. bias scan: unsteered + prompt baselines, 6 small categories x 2 models
#   -> 2. dose tests with the stereotype vector: race, then gender; Qwen, then Gemma
#   -> scan report -> results ZIP -> push it to GitHub
# A FAIL verdict is a result, not an error, so the run still finishes and packages.
# Start it with `bash run_experiment.sh run3`. Running that again resumes.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RESUME="bash run_experiment.sh run3"
source scripts/campaign_lib.sh

MODELS=(qwen gemma)
SCAN=(age religion nationality appearance disability socioeconomic)
DOSE=(race gender)
scanned=()

status "STARTED run 3"
stage run3-environment python scripts/check_gpu.py --output logs/run3-gpu-environment.json
python -m pip freeze > logs/run3-packages.txt
once run3-tests python -m pytest -q
stage run3-configs python scripts/make_run3_configs.py --check
stage run3-access python scripts/check_model_access.py google/gemma-2-9b-it

# 1. Bias scan: how much stereotyped answering each category has. Descriptive only:
#    these categories have 3-5 validation scenarios, too few for a dose test.
for model in "${MODELS[@]}"; do
  for axis in "${SCAN[@]}"; do
    name=run3-scan-$model-$axis
    config=configs/run3/scan_${model}_${axis}
    lock_once "$config" "$name-lock"
    once "$name" steerquant evaluate --config "$config.locked.json" --precision bf16 --split validation --resume
    scanned+=("$config.locked.json")
  done
done

# 2. Dose tests: Race (new category) first, then Gender (same data as run 2, new vector).
for axis in "${DOSE[@]}"; do
  for model in "${MODELS[@]}"; do
    name=run3-dose-$model-$axis
    config=configs/run3/dose_${model}_${axis}
    stage "$name-plan" steerquant plan --config "$config.yaml"
    lock_once "$config" "$name-lock"
    extract_once "$config" "artifacts/run3/dose-$model-$axis" bf16 "$name"
    once "$name-validation" steerquant evaluate --config "$config.locked.json" --precision bf16 --split validation --resume
    summarize_once "$config" validation "reports/$name" "$name-report"
    stage "$name-verdict" steerquant dose-check --config "$config.locked.json" --output "reports/$name/verdict.json"
    scanned+=("$config.locked.json")
  done
done

# The scan table covers all eight categories: the dose sweeps include the same two baselines.
stage run3-scan-report steerquant bias-scan --config "${scanned[@]}" --output reports/run3-scan/scan.json

finish run3 "Verdicts: reports/run3-dose-*/verdict.md. Bias scan: reports/run3-scan/scan.md" \
  --only run3 --artifact-root artifacts/run3
