# Shared by run_campaign.sh and run_dose.sh. Source it from the repository root after
# setting RESUME, the command that resumes this campaign.
mkdir -p logs logs/.done
exec 9>logs/.campaign.lock
if ! flock -n 9; then
  echo "Another campaign is already running in this folder."
  exit 1
fi
if ! command -v steerquant >/dev/null && [[ -f .venv/bin/activate ]]; then
  set +u; source .venv/bin/activate; set -u
fi

STAGE=startup

status() { echo "$(date -u +%FT%TZ) $*" | tee logs/campaign-status.txt; }
trap 'rc=$?; status "FAILED in stage $STAGE (exit $rc). See logs/$STAGE.log; fix the cause, then run: $RESUME"' ERR
trap 'status "STOPPED during stage $STAGE. Resume with: $RESUME"; exit 143' TERM INT HUP

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

summarize_once() {  # summarize_once <config> <split> <report dir> <log name>
  if [[ -f $3/report.md ]]; then return 0; fi
  if [[ -d $3 ]]; then mv "$3" "$3.partial-$(date +%s)"; fi  # an interrupted earlier attempt
  stage "$4" steerquant summarize --config "$1.locked.json" --split "$2" --output "$3"
}
