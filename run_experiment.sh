#!/usr/bin/env bash
# The one command for the whole study on the Ubuntu GPU machine. Everything runs inside
# a tmux session, so it keeps going after you disconnect or turn off your laptop.
#
#   bash run_experiment.sh           first time: installs what is missing, then starts the experiment.
#                                    later: resumes after a stop, failure or reboot (finished steps are
#                                    skipped), or just re-opens the view if it is still running.
#   bash run_experiment.sh status    current step, or DONE / FAILED / STOPPED
#   bash run_experiment.sh attach    open the live view (leave it with Ctrl+B, then D)
#   bash run_experiment.sh stop      stop it; finished work is kept, run again to resume
#
# Optional settings go in front, e.g. `SKIP_GATE=1 bash run_experiment.sh`:
#   HF_HOME          where the ~15 GB model is cached (default ~/.cache/huggingface)
#   HF_TOKEN         Hugging Face token; not required (the model is public)
#   MAX_INVALID, MAX_TRUNCATION, SKIP_GATE   GPU-check gate settings (GPU_RUNBOOK.md section 3)
#   QA_FILE          capability QA file (default data/capability/qa.jsonl; empty to skip)
#   TORCH_INDEX_URL  PyTorch wheel index (default CUDA 12.8 builds, needed for Blackwell)
set -euo pipefail
cd "$(dirname "$0")"
SESSION=steerquant
mkdir -p logs

running() { ! flock -n logs/.experiment.lock true; }

inside() {  # runs inside tmux: install Python packages if needed, then the campaign
  exec 8>logs/.experiment.lock
  if ! flock -n 8; then echo "The experiment is already running."; exit 1; fi
  set +u; source .venv/bin/activate; set -u
  echo "===== $(date -u +%FT%TZ) experiment start/resume ====="
  if ! python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' 2>/dev/null; then
    echo "Installing PyTorch with CUDA..."
    python -m pip install --upgrade pip
    python -m pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  fi
  local stamp
  stamp=$(sha256sum pyproject.toml | cut -d' ' -f1)
  if ! command -v steerquant >/dev/null || [[ $(cat .venv/.steerquant-installed 2>/dev/null) != "$stamp" ]]; then
    echo "Installing the project and its dependencies..."
    python -m pip install -e '.[dev,quant]'
    python -m pip check || echo "Warning: pip check reported the issues above."
    echo "$stamp" > .venv/.steerquant-installed
  fi
  local runner=(bash scripts/run_campaign.sh)
  if command -v systemd-inhibit >/dev/null && systemd-inhibit --what=sleep:idle --who=steerquant --why=probe true >/dev/null 2>&1; then
    runner=(systemd-inhibit --what=sleep:idle --who=steerquant --why="steering experiment running" "${runner[@]}")
  else
    echo "Note: could not block system suspend; make sure automatic suspend is off (GPU_RUNBOOK.md section 0)."
  fi
  "${runner[@]}"
}

case "${1:-start}" in
  __window)  # what the tmux window runs: the experiment, logged, then keep the window open
    trap ':' INT  # Ctrl+C stops the experiment, not this wrapper (children still get the default)
    set +e
    bash run_experiment.sh __inside 2>&1 | tee -i -a logs/campaign.log
    echo
    echo "Finished. Status: $(cat logs/campaign-status.txt 2>/dev/null)"
    read -r -p "Press Enter to close this window. "
    exit 0 ;;
  __inside)
    inside
    exit ;;
  status)
    cat logs/campaign-status.txt 2>/dev/null || echo "Not started yet."
    if running; then echo "(running in tmux session '$SESSION')"; else echo "(not running)"; fi
    exit 0 ;;
  attach)
    exec tmux attach -t "$SESSION" ;;
  stop)
    if ! running; then echo "Not running."; exit 0; fi
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 60); do if running; then sleep 1; else break; fi; done
    if running; then tmux kill-session -t "$SESSION"; fi
    echo "Stopped. Last status: $(cat logs/campaign-status.txt 2>/dev/null)"
    exit 0 ;;
  start) ;;
  *)
    echo "Usage: bash run_experiment.sh [start|status|attach|stop]"
    exit 1 ;;
esac

if running; then
  echo "The experiment is already running. Opening the view (leave it with Ctrl+B, then D)."
  if [[ -t 0 && -z ${TMUX:-} ]]; then exec tmux attach -t "$SESSION"; fi
  exit 0
fi

# One-time system prerequisites. apt-get may ask for your sudo password.
if [[ $(uname -s) != Linux ]] || ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null; then
  echo "This needs Linux with a working NVIDIA driver (nvidia-smi failed)."
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "Python 3.10 or newer is required."
  exit 1
fi
if ! command -v tmux >/dev/null; then
  echo "Installing tmux..."
  sudo apt-get update && sudo apt-get install -y tmux
fi
if [[ ! -x .venv/bin/python ]]; then
  rm -rf .venv
  if ! python3 -m venv .venv; then
    echo "Installing python3-venv..."
    sudo apt-get install -y python3-venv
    rm -rf .venv
    python3 -m venv .venv
  fi
fi
cache=${HF_HOME:-$HOME/.cache/huggingface}
probe=$cache
while [[ ! -d $probe ]]; do probe=$(dirname "$probe"); done
free_gb=$(( $(df -Pk "$probe" | awk 'NR == 2 {print $4}') / 1024 / 1024 ))
if (( free_gb < 50 )); then
  echo "Warning: only ${free_gb} GB free for the model cache ($cache); about 50 GB is recommended."
fi

# A finished, failed or stopped earlier session is replaced; the campaign resumes from its records.
tmux kill-session -t "$SESSION" 2>/dev/null || true
settings=(-e "QA_FILE=${QA_FILE-data/capability/qa.jsonl}")
for var in HF_HOME HF_TOKEN MAX_INVALID MAX_TRUNCATION SKIP_GATE TORCH_INDEX_URL; do
  if [[ -n ${!var:-} ]]; then settings+=(-e "$var=${!var}"); fi
done
tmux new-session -d -s "$SESSION" -c "$PWD" "${settings[@]}" "bash run_experiment.sh __window"
echo "Started in tmux session '$SESSION'. Leave the view with Ctrl+B, then D: the experiment keeps running."
echo "Later, from any terminal on this machine:  bash run_experiment.sh status | attach | stop"
sleep 2
if [[ -t 0 && -z ${TMUX:-} ]]; then exec tmux attach -t "$SESSION"; fi
