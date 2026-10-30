#!/usr/bin/env bash
# Runs the whole pilot unattended, in this order:
#   environment check -> offline tests -> GPU check (BF16/INT8/NF4) -> go/no-go gate
#   -> main validation sweep -> frozen selection -> held-out test -> [capability] -> results ZIP
# Finished stages are detected and skipped, so running it again resumes after a
# crash, reboot or power cut. Launch it with scripts/start_campaign.sh, which
# detaches it from the terminal so it survives logging out.
#
# Optional environment variables:
#   MAX_INVALID, MAX_TRUNCATION  gate limits for the UNSTEERED baseline (default 0.05 each)
#   SKIP_GATE=1                  continue although the gate failed (only after reading logs/gate.log)
#   QA_FILE=/path/to/qa.jsonl    also run the held-out capability/neutral-uncertainty evaluation
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs logs/.done
exec 9>logs/.campaign.lock
if ! flock -n 9; then
  echo "Another campaign is already running in this folder."
  exit 1
fi
if ! command -v steerquant >/dev/null && [[ -f .venv/bin/activate ]]; then
  set +u; source .venv/bin/activate; set -u
fi

CHECK=configs/qwen_gender_gpu_check
MAIN=configs/qwen_gender
CHECK_ROOT=artifacts/gpu-check
MAIN_ROOT=artifacts/qwen-gender-seed42
PRECISIONS=(bf16 int8 nf4)
STAGE=startup

status() { echo "$(date -u +%FT%TZ) $*" | tee logs/campaign-status.txt; }
trap 'rc=$?; status "FAILED in stage $STAGE (exit $rc). See logs/$STAGE.log; fix the cause, then run scripts/start_campaign.sh to resume."' ERR
trap 'status "STOPPED by request during stage $STAGE. Run scripts/start_campaign.sh to resume."; exit 143' TERM INT

stage() {  # stage <name> <command...>: run and log to logs/<name>.log
  STAGE=$1; shift
  status "RUNNING $STAGE"
  "$@" 2>&1 | tee -a "logs/$STAGE.log"
}

once() {  # once <name> <command...>: like stage, but skipped after it has succeeded once
  if [[ -f logs/.done/$1 ]]; then echo "skip $1 (already done)"; return 0; fi
  stage "$@"
  touch "logs/.done/$1"
}

lock_once() {  # lock_once <config without extension>
  if [[ ! -f $1.locked.json ]]; then
    stage "$(basename "$1")-lock" steerquant lock --config "$1.yaml" --output "$1.locked.json"
  fi
}

extract_once() {  # extract_once <config> <artifact root> <precision> <log prefix>
  if [[ ! -f $2/vectors/$3.pt ]]; then
    stage "$4-$3-extract" steerquant extract --config "$1.locked.json" --precision "$3"
  fi
}

select_once() {  # select_once <config> <selection file> <log name>
  if [[ ! -f $2 ]]; then
    stage "$3" steerquant select --config "$1.locked.json" --output "$2" \
      --accuracy-budget 0.02 --invalid-budget 0.01 --unknown-budget 0.02
  fi
}

summarize_once() {  # summarize_once <config> <split> <report dir> <log name>
  if [[ -f $3/report.md ]]; then return 0; fi
  if [[ -d $3 ]]; then mv "$3" "$3.partial-$(date +%s)"; fi  # an interrupted earlier attempt
  stage "$4" steerquant summarize --config "$1.locked.json" --split "$2" --output "$3"
}

status "STARTED"

# 1. Environment and offline checks.
stage environment python scripts/check_gpu.py --output logs/gpu_environment.json
python -m pip freeze > logs/gpu-packages.txt
once tests python -m pytest -q
if [[ ! -f logs/.done/offline-smoke && -d artifacts/offline-smoke ]]; then
  mv artifacts/offline-smoke "artifacts/offline-smoke.partial-$(date +%s)"
fi
once offline-smoke steerquant smoke --output artifacts/offline-smoke
stage main-plan steerquant plan --config "$MAIN.yaml"

# 2. Small GPU check on every precision and code path (validation subset only).
stage gpu-check-plan steerquant plan --config "$CHECK.yaml"
lock_once "$CHECK"
for p in "${PRECISIONS[@]}"; do extract_once "$CHECK" "$CHECK_ROOT" "$p" gpu-check; done
for p in "${PRECISIONS[@]}"; do
  once "gpu-check-$p-validation" steerquant evaluate --config "$CHECK.locked.json" --precision "$p" --split validation --resume
done
summarize_once "$CHECK" validation reports/gpu-check gpu-check-report
select_once "$CHECK" "$CHECK_ROOT/selection.json" gpu-check-selection  # exercises selection only

# 3. Go/no-go gate before the long campaign.
STAGE=gate
status "RUNNING gate"
if ! python scripts/check_gate.py --config "$CHECK.locked.json" --output logs/gate.json \
     --max-invalid "${MAX_INVALID:-0.05}" --max-truncation "${MAX_TRUNCATION:-0.05}" 2>&1 | tee logs/gate.log; then
  if [[ ${SKIP_GATE:-0} == 1 ]]; then
    echo "Gate failed, continuing because SKIP_GATE=1" | tee -a logs/gate.log
  else
    status "STOPPED at the GPU-check gate. Read logs/gate.log and GPU_RUNBOOK.md section 3."
    exit 1
  fi
fi

# 4. Main campaign: vectors and full validation sweep, one precision at a time.
lock_once "$MAIN"
for p in "${PRECISIONS[@]}"; do
  extract_once "$MAIN" "$MAIN_ROOT" "$p" main
  once "main-$p-validation" steerquant evaluate --config "$MAIN.locked.json" --precision "$p" --split validation --resume
done
summarize_once "$MAIN" validation reports/gender-validation main-validation-report

# 5. Freeze settings on validation, then evaluate the held-out test split once.
select_once "$MAIN" "$MAIN_ROOT/selection.json" main-selection
once main-test steerquant evaluate --config "$MAIN.locked.json" --precision all --split test \
  --selection "$MAIN_ROOT/selection.json" --resume
summarize_once "$MAIN" test reports/gender-test main-test-report

# 6. Optional held-out capability / neutral-uncertainty suite at the same frozen settings.
if [[ -n ${QA_FILE:-} ]]; then
  once main-capability steerquant evaluate --config "$MAIN.locked.json" --precision all --split capability \
    --qa-file "$QA_FILE" --selection "$MAIN_ROOT/selection.json" --resume
  summarize_once "$MAIN" capability reports/gender-capability main-capability-report
fi

# 7. Package everything to send back.
ZIP="steering-results-$(date -u +%Y%m%dT%H%M%SZ).zip"
stage package python scripts/package_results.py --output "$ZIP"
status "DONE. Results archive: $ZIP"
