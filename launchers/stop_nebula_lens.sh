#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${NEBULA_LENS_PORT:-8501}"
DESKTOP_PID_FILE="$ROOT_DIR/.nebula_lens.pid"
STREAMLIT_PID_FILE="$ROOT_DIR/.nebula_lens_streamlit.pid"

show_dialog() {
  local message="$1"
  local escaped_message
  escaped_message="$(printf '%s' "$message" | /usr/bin/sed 's/\\/\\\\/g; s/"/\\"/g')"
  /usr/bin/osascript >/dev/null 2>&1 <<OSA || true
display dialog "${escaped_message}" buttons {"OK"} default button "OK"
OSA
}

typeset -A unique_pids

for pid_file in "$DESKTOP_PID_FILE" "$STREAMLIT_PID_FILE"; do
  if [[ ! -f "$pid_file" ]]; then
    continue
  fi

  recorded_pid="$(<"$pid_file")"
  if [[ -n "$recorded_pid" ]] && /bin/kill -0 "$recorded_pid" >/dev/null 2>&1; then
    unique_pids["$recorded_pid"]=1
  fi
done

for pid in "${(@f)$(/usr/sbin/lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)}"; do
  [[ -n "$pid" ]] && unique_pids["$pid"]=1
done

pids=("${(@k)unique_pids}")

if (( ${#pids[@]} == 0 )); then
  /bin/rm -f "$DESKTOP_PID_FILE" "$STREAMLIT_PID_FILE"
  show_dialog "Nebula Lens is not running."
  exit 0
fi

/bin/kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
/bin/sleep 1

remaining=("${(@f)$(/usr/sbin/lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)}")
if (( ${#remaining[@]} > 0 )); then
  /bin/kill -KILL "${remaining[@]}" >/dev/null 2>&1 || true
fi

/bin/rm -f "$DESKTOP_PID_FILE" "$STREAMLIT_PID_FILE"

show_dialog "Nebula Lens was stopped."
