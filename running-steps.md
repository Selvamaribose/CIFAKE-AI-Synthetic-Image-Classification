# Nebula Lens Setup Guide

## Quick Setup

### macOS

```zsh
git clone https://github.com/Selvamaribose/CIFAKE-AI-Synthetic-Image-Classification.git
cd CIFAKE-AI-Synthetic-Image-Classification

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x launchers/*.sh launchers/*.command
./launchers/build_nebula_lens_app.sh
```

Then open:

- `Nebula Lens.app` from the repo root, or
- `Quick Start/Nebula Lens.app`

### Windows

```bat
git clone https://github.com/Selvamaribose/CIFAKE-AI-Synthetic-Image-Classification.git
cd CIFAKE-AI-Synthetic-Image-Classification

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then open:

- `Quick Start\Open Nebula Lens.bat`

## What Opens

- The desktop wrapper lives in `desktop_app.py`
- It starts the local Streamlit server on `127.0.0.1:8501`
- It opens the UI inside a native desktop window by using `pywebview`
- Closing the desktop window normally also stops the local server

## If The App Hangs

Use the stop launcher once, then reopen:

- macOS: `Quick Start/Stop Nebula Lens.command`
- Windows: `Quick Start/Stop Nebula Lens.bat`

## Browser Fallback

If you want the old browser flow for debugging, you can still run:

```zsh
source .venv/bin/activate
python3 -m streamlit run deploy.py
```

## Retraining

Only retrain when you actually want a new model.

### Download datasets once

```zsh
python3 prepare_datasets.py --download --build-manifest --continue-on-error --sources \
  birdy654_cifake_real_and_ai_generated_synthetic_images \
  cashbowman_ai_generated_images_vs_real_images \
  swati6945_ai_generated_vs_real_images
```

### Rebuild the manifest later without redownloading

```zsh
python3 prepare_datasets.py --build-manifest --continue-on-error
```

### Train

```zsh
python3 train_model.py --manifest datasets/dataset_manifest.json
```

## Notes

- `AIGeneratedModel.weights.h5` is the inference file used by the app
- `training_metrics.json` stores the latest evaluation summary
- `prepare_datasets.py` downloads datasets only once in normal use
