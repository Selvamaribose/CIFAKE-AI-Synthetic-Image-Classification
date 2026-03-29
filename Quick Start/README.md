# Quick Start

Open from this folder:

- macOS: double-click `Nebula Lens.app`
- macOS fallback: double-click `Open Nebula Lens.command`
- macOS stop: double-click `Stop Nebula Lens.command`
- Windows: double-click `Open Nebula Lens.bat`
- Windows stop: double-click `Stop Nebula Lens.bat`

Nebula Lens now opens in its own native desktop window by using the repo's `desktop_app.py` wrapper.
In the normal path, closing that window also stops the local Streamlit server.

If the app ever hangs, use one of the stop launchers, then open it again.

The underlying launcher scripts live in the repo's `launchers` folder, so the project root stays cleaner.
