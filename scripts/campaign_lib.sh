# Shared by run_campaign.sh, run_dose.sh and run_run3.sh. Source it from the repository root after
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

lock_once() {  # lock_once <config without extension> [log name]
  if [[ ! -f $1.locked.json ]]; then
    stage "${2:-$(basename "$1")-lock}" steerquant lock --config "$1.yaml" --output "$1.locked.json"
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

push_results() {  # push_results <zip>: commit only this archive and push it to the current branch
  local zip=$1 size attempt
  export GIT_TERMINAL_PROMPT=0  # fail at once instead of waiting for a password nobody will type
  size=$(stat -c %s "$zip")
  if (( size > 95 * 1024 * 1024 )); then
    echo "$zip is $size bytes, over GitHub's 100 MB file limit; not pushed."
    return 1
  fi
  git add -- "$zip" || return 1
  if ! git diff --cached --quiet -- "$zip"; then
    git commit -m "Results: $zip" -- "$zip" || return 1  # only this file, whatever else is staged
  fi
  for attempt in 1 2 3; do
    if git push; then return 0; fi
    echo "Push attempt $attempt failed; updating from the remote and retrying."
    git pull --rebase --autostash || true
    sleep $((attempt * ${PUSH_RETRY_SECONDS:-20}))
  done
  return 1
}

finish() {  # finish <name> <done message> <package_results.py arguments...>
  # Package once (a resumed, finished campaign reuses its archive), then push it unless
  # PUSH_RESULTS=0. A failed push never fails the campaign: the results are safe on disk.
  local name=$1 message=$2 zip
  shift 2
  if [[ ! -f logs/.done/$name-package ]]; then
    zip="$name-results-$(date -u +%Y%m%dT%H%M%SZ).zip"
    stage "$name-package" python scripts/package_results.py --output "$zip" "$@"
    echo "$zip" > "logs/.done/$name-package"
  fi
  zip=$(cat "logs/.done/$name-package")
  if [[ ${PUSH_RESULTS:-1} == 0 ]]; then
    status "DONE. $message. Results archive: $zip (not pushed: PUSH_RESULTS=0)"
    return 0
  fi
  STAGE=$name-push
  status "RUNNING $name-push"
  if push_results "$zip" 2>&1 | tee -a "logs/$name-push.log"; then
    status "DONE. $message. Results archive: $zip, pushed to GitHub"
  else
    status "DONE, but the results archive was NOT pushed (see logs/$name-push.log). $message. Push it by hand: git add $zip && git commit -m \"Results: $zip\" -- $zip && git push"
  fi
}
