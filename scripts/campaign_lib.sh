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

read_setting() {  # read_setting <name> <file...>: the value of name=value in the first file that has it
  local name=$1 file line value
  shift
  for file in "$@"; do
    [[ -f $file ]] || continue
    while IFS= read -r line || [[ -n $line ]]; do
      line=${line%$'\r'}
      [[ $line == *=* ]] || continue
      [[ $(printf '%s' "${line%%=*}" | tr -d '[:space:]') == "$name" ]] || continue
      value=${line#*=}
      if [[ $value =~ ^[[:space:]]*\"(.*)\"[[:space:]]*$ || $value =~ ^[[:space:]]*\'(.*)\'[[:space:]]*$ ]]; then
        value=${BASH_REMATCH[1]}  # quoted: keep exactly what is inside the quotes
      else
        value=${value#"${value%%[![:space:]]*}"}; value=${value%"${value##*[![:space:]]}"}
      fi
      printf '%s' "$value"
      return 0
    done < "$file"
  done
  return 1
}

unlock_ssh_key() {  # for an SSH remote, load the key into a private ssh-agent; never prompts
  # The passphrase is github_passkey in steerquant.env or .env (both git-ignored). It is
  # handed to ssh-add through a one-shot askpass helper, never on a command line or in a
  # log. A wrong passphrase fails at once instead of ssh-add asking again forever.
  local url key pass dir
  url=$(git remote get-url origin 2>/dev/null) || return 0
  [[ $url == git@* || $url == ssh://* ]] || return 0  # HTTPS remotes use the credential helper
  key=${GITHUB_SSH_KEY:-$HOME/.ssh/id_ed25519}
  if [[ ! -f $key ]]; then echo "SSH key $key not found (set GITHUB_SSH_KEY)."; return 1; fi
  pass=$(read_setting github_passkey steerquant.env .env || read_setting GITHUB_PASSKEY steerquant.env .env || true)
  if [[ -z $pass ]]; then echo "No github_passkey in steerquant.env or .env; trying $key without a passphrase."; fi
  eval "$(ssh-agent -s)" >/dev/null || return 1
  OWN_AGENT=$SSH_AGENT_PID  # stop only this agent afterwards, never one the user started
  dir=$(mktemp -d) && chmod 700 "$dir" || return 1
  # Answers once; a second request (bad passphrase) gets nothing, so ssh-add gives up.
  printf '%s\n' '#!/bin/sh' "[ -e \"$dir/used\" ] && exit 1" ": > \"$dir/used\"" \
    'printf "%s\n" "$STEERQUANT_KEY_PASSPHRASE"' > "$dir/askpass"
  chmod 700 "$dir/askpass"
  if ! STEERQUANT_KEY_PASSPHRASE=$pass SSH_ASKPASS=$dir/askpass SSH_ASKPASS_REQUIRE=force DISPLAY=${DISPLAY:-none} \
       timeout 60 ssh-add "$key" </dev/null; then
    rm -rf "$dir"
    echo "Could not unlock $key: check github_passkey."
    return 1
  fi
  rm -rf "$dir"
  # Fail instead of prompting if the agent cannot authenticate; trust GitHub's key on first use only.
  export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
}

stop_own_agent() {
  if [[ -n ${OWN_AGENT:-} ]]; then SSH_AGENT_PID=$OWN_AGENT ssh-agent -k >/dev/null 2>&1 || true; OWN_AGENT=; fi
}

push_results() {  # push_results <zip>: commit only this archive and push it to the current branch
  local zip=$1 size attempt pushed=1
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
  OWN_AGENT=
  if ! unlock_ssh_key; then stop_own_agent; return 1; fi
  for attempt in 1 2 3; do
    if git push; then pushed=0; break; fi
    echo "Push attempt $attempt failed; updating from the remote and retrying."
    git pull --rebase --autostash || true
    sleep $((attempt * ${PUSH_RETRY_SECONDS:-20}))
  done
  stop_own_agent
  return $pushed
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
