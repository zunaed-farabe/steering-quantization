#!/usr/bin/env bash
# Start (or resume) the whole campaign in the background, detached from this
# terminal. It keeps running after you close SSH, log out, or turn off your laptop.
#
#   bash scripts/start_campaign.sh     start, or resume after a failure/reboot
#   cat logs/campaign-status.txt       current stage, or DONE / FAILED / STOPPED
#   tail -f logs/campaign.log          live progress (Ctrl+C stops watching, not the job)
#   bash scripts/stop_campaign.sh      stop it; completed work is kept
#
# Settings for scripts/run_campaign.sh pass through, e.g.
#   QA_FILE=/data/held_out_qa.jsonl bash scripts/start_campaign.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [[ -z ${VIRTUAL_ENV:-} && -f .venv/bin/activate ]]; then
  set +u; source .venv/bin/activate; set -u
fi
if ! command -v steerquant >/dev/null; then
  echo "steerquant is not installed in the active Python environment; see GPU_RUNBOOK.md section 1."
  exit 1
fi
if ! flock -n logs/.campaign.lock true; then
  echo "A campaign is already running here. Watch it with: tail -f logs/campaign.log"
  exit 1
fi

runner=(bash scripts/run_campaign.sh)
# Block system suspend while the job runs, when this user is allowed to.
if command -v systemd-inhibit >/dev/null && systemd-inhibit --what=sleep:idle --who=steerquant --why=probe true >/dev/null 2>&1; then
  runner=(systemd-inhibit --what=sleep:idle --who=steerquant --why="steering experiment running" "${runner[@]}")
else
  echo "Note: could not block system suspend automatically; turn automatic suspend off (GPU_RUNBOOK.md section 0)."
fi

echo "===== launched $(date -u +%FT%TZ) =====" >> logs/campaign.log
# setsid: own session, so closing the terminal/SSH sends it no hangup; nohup as a second guard.
nohup setsid "${runner[@]}" >> logs/campaign.log 2>&1 < /dev/null &
echo $! > logs/campaign.pid
sleep 3
if kill -0 "$(cat logs/campaign.pid)" 2>/dev/null; then
  echo "Campaign running in the background (process group $(cat logs/campaign.pid))."
  echo "You can now close this terminal or turn off your laptop."
  echo "Status:   cat logs/campaign-status.txt"
  echo "Progress: tail -f logs/campaign.log"
else
  echo "The campaign exited immediately. Last log lines:"
  tail -n 20 logs/campaign.log
  exit 1
fi
