#!/usr/bin/env bash
# Stop a campaign started with scripts/start_campaign.sh. Completed runs are kept;
# running start_campaign.sh again resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f logs/campaign.pid ]]; then
  echo "No logs/campaign.pid: no campaign was started from this folder."
  exit 1
fi
pid=$(cat logs/campaign.pid)
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid"  # the whole process group, including the running Python job
  echo "Stop signal sent to campaign process group $pid."
else
  echo "The campaign is not running. Last status: $(cat logs/campaign-status.txt 2>/dev/null || echo unknown)"
fi
