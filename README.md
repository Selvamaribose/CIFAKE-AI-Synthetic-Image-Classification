# AI Image Classifier

This project is a Streamlit app for classifying images as `Real` or `AI Generated`.
It also includes:

- a reusable training pipeline
- optional Kaggle dataset download + manifest generation
- a saved model weights file for inference
- a desktop wrapper that opens the app in its own native window on macOS and Windows
- confidence breakdown and visual explanation in the app

## Repository Contents

- `deploy.py`: Streamlit application
- `train_model.py`: model training script
- `prepare_datasets.py`: Kaggle dataset downloader and manifest builder
- `model_utils.py`: shared model configuration
- `desktop_app.py`: native desktop wrapper for the Streamlit UI
- `AIGeneratedModel.weights.h5`: trained inference weights
- `training_metrics.json`: latest saved training/evaluation summary

## Quick Start

If `AIGeneratedModel.weights.h5` is already in the repository, you do not need to train anything.

### macOS / Linux

```zsh
git clone https://github.com/Selvamaribose/CIFAKE-AI-Synthetic-Image-Classification.git
cd CIFAKE-AI-Synthetic-Image-Classification

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run deploy.py
```

If `streamlit` is not found, use:

```zsh
python3 -m streamlit run deploy.py
```

### Windows PowerShell

```powershell
git clone https://github.com/Selvamaribose/CIFAKE-AI-Synthetic-Image-Classification.git
cd CIFAKE-AI-Synthetic-Image-Classification

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

streamlit run deploy.py
```

## When Training Is Needed

You only need to run training if:

- the repository does not include `AIGeneratedModel.weights.h5`
- you added new datasets
- you changed the training code
- you want a better model

Training is slow and is not required every time you start the app.

## Retraining Workflow

### 1. Activate the virtual environment

macOS / Linux:

```zsh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Download datasets once

This step is only needed once per machine unless you delete the downloaded datasets.

```zsh
python3 prepare_datasets.py --download --build-manifest --continue-on-error --sources \
  birdy654_cifake_real_and_ai_generated_synthetic_images \
  cashbowman_ai_generated_images_vs_real_images \
  swati6945_ai_generated_vs_real_images
```

This creates:

- `datasets/source_index.json`
- `datasets/dataset_manifest.json`

### 3. Train the model

```zsh
python3 train_model.py --manifest datasets/dataset_manifest.json
```

This updates:

- `AIGeneratedModel.weights.h5`
- `AIGeneratedModel.keras`
- `training_metrics.json`

### 4. Run the app

```zsh
streamlit run deploy.py
```

### 5. Open it like an app

For the easiest Finder/File Explorer launch, open the `Quick Start` folder in the repo root.

macOS:

```zsh
chmod +x launchers/*.sh launchers/*.command
./launchers/build_nebula_lens_app.sh
```

This builds `Nebula Lens.app` in the project root.

How to open it on Mac:

1. Open the `Quick Start` folder
2. Double-click `Nebula Lens.app`
3. Nebula Lens starts a local Streamlit server and opens it inside its own desktop window
4. Closing the Nebula Lens window normally also stops the local server

You can also open it from the repo root:

- `Nebula Lens.app`
- `launchers/Launch Nebula Lens.command`
- `launchers/Stop Nebula Lens.command`

Windows:

How to open it on Windows:

1. Open the `Quick Start` folder
2. Double-click `Open Nebula Lens.bat`
3. Nebula Lens starts a local Streamlit server and opens it inside its own desktop window
4. Closing the Nebula Lens window normally also stops the local server

Or run it from Command Prompt:

```bat
launchers\Launch Nebula Lens.bat
```

To stop it cleanly later:

- macOS: double-click `Quick Start/Stop Nebula Lens.command`
- Windows: double-click `Quick Start/Stop Nebula Lens.bat`

Or run:

```zsh
launchers/Stop\ Nebula\ Lens.command
```

```bat
launchers\Stop Nebula Lens.bat
```

## Important Notes

- The app loads `AIGeneratedModel.weights.h5` first.
- If you commit `AIGeneratedModel.weights.h5` and `training_metrics.json`, another system can run the app without retraining.
- `prepare_datasets.py --download ...` downloads datasets only once in normal use.
- `train_model.py` takes time because it performs real training. Do not run it unless you want a new model.

## Data Sources

The current dataset workflow supports source-aware training and evaluation. The repo is configured for these Kaggle datasets:

- CIFAKE: Real and AI-Generated Synthetic Images
- cashbowman/ai-generated-images-vs-real-images
- swati6945/ai-generated-vs-real-images
- sonalraj1234/ai-vs-real-images
- muhammadsaoodsarwar/ai-vs-real-192-class-scene-image-dataset

Recommended starter set:

- `birdy654_cifake_real_and_ai_generated_synthetic_images`
- `cashbowman_ai_generated_images_vs_real_images`
- `swati6945_ai_generated_vs_real_images`

## Current App Behavior

- Accepts common image formats including `jpg`, `jpeg`, `png`, `bmp`, and `webp`
- Shows Real vs AI classification
- Shows score breakdown
- Shows a visual explanation heatmap
- Uses a saved training threshold from `training_metrics.json`

## Troubleshooting

### The Mac app or Windows launcher does not open

Make sure the desktop dependencies were installed:

```zsh
pip install -r requirements.txt
```

The native window uses `pywebview`, so the desktop entrypoints need that package in `.venv`.

### `python: command not found` on macOS

Use `python3`, not `python`.

```zsh
python3 -m venv .venv
```

### `streamlit: command not found`

Run Streamlit through Python:

```zsh
python3 -m streamlit run deploy.py
```

### The desktop window hangs or stays in the background

Use the stop launcher once, then reopen the app:

- macOS: `Quick Start/Stop Nebula Lens.command`
- Windows: `Quick Start/Stop Nebula Lens.bat`

### Kaggle download timeout

Start with the smaller subset shown above instead of downloading every configured dataset at once.

### The app misclassifies some real portrait images

That usually means the training data needs better balance, especially more real human portraits matched against AI-generated portraits. Retrain after improving the dataset mix.

## Model Files

- `AIGeneratedModel.weights.h5`: inference weights used by the app
- `AIGeneratedModel.keras`: best saved Keras checkpoint from training
- `AIGeneratedModel.h5`: legacy model file kept for backward compatibility

## Evaluation

The latest evaluation summary is saved in `training_metrics.json`.
Check that file for:

- dataset counts
- split counts
- source-wise metrics
- decision threshold
- latest test metrics
