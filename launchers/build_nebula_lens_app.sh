#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="Nebula Lens.app"
APP_PATH="$ROOT_DIR/$APP_NAME"
ICON_SOURCE="$ROOT_DIR/app_icon/Nebula-Icon.png"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
MACOS_DIR="$APP_PATH/Contents/MacOS"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
INFO_PLIST="$APP_PATH/Contents/Info.plist"
APP_LAUNCHER="$MACOS_DIR/Nebula Lens"
TMP_DIR="$(mktemp -d)"
ICNS_PATH="$TMP_DIR/NebulaIcon.icns"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" - "$ICON_SOURCE" "$ICNS_PATH" <<'PY'
from pathlib import Path
import sys

from PIL import Image

source = Path(sys.argv[1])
target = Path(sys.argv[2])
image = Image.open(source).convert("RGBA")
sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
image.save(target, format="ICNS", sizes=sizes)
PY

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cat > "$INFO_PLIST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>Nebula Lens</string>
  <key>CFBundleIconFile</key>
  <string>NebulaIcon.icns</string>
  <key>CFBundleIdentifier</key>
  <string>com.pavithra.nebulalens</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Nebula Lens</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
</dict>
</plist>
PLIST

cat > "$APP_LAUNCHER" <<'LAUNCHER'
#!/bin/zsh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
exec "$REPO_DIR/launchers/launch_nebula_lens.sh"
LAUNCHER

/bin/chmod +x "$APP_LAUNCHER"
/bin/cp "$ICNS_PATH" "$RESOURCES_DIR/NebulaIcon.icns"

echo "Built $APP_PATH"
