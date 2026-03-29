#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python3"
DESKTOP_ENTRY="$ROOT_DIR/desktop_app.py"

show_dialog() {
  local message="$1"
  local escaped_message
  escaped_message="$(printf '%s' "$message" | /usr/bin/sed 's/\\/\\\\/g; s/"/\\"/g')"
  /usr/bin/osascript >/dev/null 2>&1 <<OSA || true
display dialog "${escaped_message}" buttons {"OK"} default button "OK" with icon caution
OSA
}

if [[ ! -x "$VENV_PYTHON" ]]; then
  show_dialog "Nebula Lens could not find .venv/bin/python3. Create the virtual environment first, then try again."
  exit 1
fi

if [[ ! -f "$DESKTOP_ENTRY" ]]; then
  show_dialog "Nebula Lens could not find desktop_app.py in the repository root."
  exit 1
fi

if ! "$VENV_PYTHON" -c "import streamlit, webview" >/dev/null 2>&1; then
  show_dialog "Nebula Lens needs the desktop dependencies first. Run 'pip install -r requirements.txt' once, then open the app again."
  exit 1
fi

exec "$VENV_PYTHON" "$DESKTOP_ENTRY"
