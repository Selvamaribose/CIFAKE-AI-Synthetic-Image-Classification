from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


APP_TITLE = "Nebula Lens"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 980
WINDOW_MIN_SIZE = (1100, 760)
STARTUP_TIMEOUT_SECONDS = 45

ROOT_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("NEBULA_LENS_PORT", "8501"))
URL = f"http://127.0.0.1:{PORT}"
LOG_FILE = ROOT_DIR / ".nebula_lens_streamlit.log"
DESKTOP_PID_FILE = ROOT_DIR / ".nebula_lens.pid"
STREAMLIT_PID_FILE = ROOT_DIR / ".nebula_lens_streamlit.pid"


def escape_applescript(message: str) -> str:
    return message.replace("\\", "\\\\").replace('"', '\\"')


def show_message(message: str, *, title: str = APP_TITLE, error: bool = False) -> None:
    if sys.platform == "darwin":
        icon = "caution" if error else "note"
        script = (
            f'display dialog "{escape_applescript(message)}" '
            f'with title "{escape_applescript(title)}" '
            f'buttons {{"OK"}} default button "OK" with icon {icon}'
        )
        subprocess.run(["osascript", "-e", script], check=False)
        return

    if sys.platform == "win32":
        try:
            import ctypes

            flags = 0x10 if error else 0x40
            ctypes.windll.user32.MessageBoxW(0, message, title, flags)
            return
        except Exception:
            pass

    stream = sys.stderr if error else sys.stdout
    print(f"{title}: {message}", file=stream)


def read_pid(path: Path) -> Optional[int]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None

    return int(value) if value.isdigit() else None


def write_pid(path: Path, pid: int) -> None:
    path.write_text(f"{pid}\n", encoding="utf-8")


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def is_pid_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_server_ready() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=1) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def wait_for_server() -> bool:
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_server_ready():
            return True
        time.sleep(1)
    return False


def start_streamlit() -> tuple[subprocess.Popen[bytes], object]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "deploy.py",
        "--server.port",
        str(PORT),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    log_handle = open(LOG_FILE, "ab")
    popen_kwargs: dict[str, object] = {
        "cwd": str(ROOT_DIR),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    write_pid(STREAMLIT_PID_FILE, process.pid)
    return process, log_handle


def stop_process(process: Optional[subprocess.Popen[bytes]]) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class DesktopApp:
    def __init__(self) -> None:
        self.streamlit_process: Optional[subprocess.Popen[bytes]] = None
        self.streamlit_log_handle = None
        self.owns_streamlit = False
        self.cleaned_up = False

    def cleanup(self) -> None:
        if self.cleaned_up:
            return

        self.cleaned_up = True

        if self.owns_streamlit:
            stop_process(self.streamlit_process)

        if self.streamlit_log_handle is not None:
            try:
                self.streamlit_log_handle.close()
            except OSError:
                pass

        if read_pid(DESKTOP_PID_FILE) == os.getpid():
            remove_file(DESKTOP_PID_FILE)

        if self.owns_streamlit:
            remove_file(STREAMLIT_PID_FILE)

    def on_window_closed(self, *_args: object) -> None:
        self.cleanup()

    def ensure_single_app_instance(self) -> bool:
        existing_pid = read_pid(DESKTOP_PID_FILE)
        if existing_pid is None:
            return True

        if existing_pid == os.getpid():
            return True

        if is_pid_running(existing_pid):
            show_message(
                "Nebula Lens is already open. Close the current app window or use the Stop launcher first."
            )
            return False

        remove_file(DESKTOP_PID_FILE)
        return True

    def ensure_streamlit_server(self) -> bool:
        if is_server_ready():
            self.owns_streamlit = False
            return True

        try:
            self.streamlit_process, self.streamlit_log_handle = start_streamlit()
        except OSError as exc:
            show_message(f"Could not start the local Streamlit server.\n\n{exc}", error=True)
            return False

        self.owns_streamlit = True
        if wait_for_server():
            return True

        self.cleanup()
        show_message(
            "Nebula Lens could not finish starting the local app server.\n\n"
            "Check .nebula_lens_streamlit.log for details.",
            error=True,
        )
        return False

    def run(self) -> int:
        try:
            import webview
        except ImportError:
            show_message(
                "The desktop wrapper needs pywebview.\n\nRun 'pip install -r requirements.txt' once, then open Nebula Lens again.",
                error=True,
            )
            return 1

        if not self.ensure_single_app_instance():
            return 0

        if not self.ensure_streamlit_server():
            return 1

        write_pid(DESKTOP_PID_FILE, os.getpid())
        atexit.register(self.cleanup)

        window = webview.create_window(
            APP_TITLE,
            URL,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            min_size=WINDOW_MIN_SIZE,
            background_color="#08101f",
            text_select=True,
        )
        window.events.closed += self.on_window_closed

        try:
            webview.start()
        finally:
            self.cleanup()

        return 0


def main() -> int:
    return DesktopApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
