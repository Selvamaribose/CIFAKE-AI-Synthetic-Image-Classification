# AI Image Classifier - Setup Guide

## Prerequisites
- Python 3.12 (compatible version)
- Git

## Installation Steps

### For Windows (Command Prompt)

#### 1. Clone the Repository
cmd
git init
git clone https://github.com/SanKolisetty/AI-Image-Classifier.git
cd AI-Image-Classifier


#### 2. Install Compatible Python Version
- Download Python 3.12 from [python.org](https://www.python.org/downloads/)
- Install and make sure to check "Add Python to PATH"

#### 3. Create Virtual Environment
cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate


#### 4. Install Dependencies
cmd
# Install all required packages
pip install -r requirements.txt

# Fix numpy compatibility if needed
pip install "numpy>=1.26.0,<2.0" --force-reinstall


#### 5. Run the Application
cmd
# Start the Streamlit app
streamlit run deploy.py


### For macOS/Linux (Terminal)

#### 1. Clone the Repository
bash
git init
git clone https://github.com/SanKolisetty/AI-Image-Classifier.git
cd AI-Image-Classifier


#### 2. Install Compatible Python Version
bash
# macOS with Homebrew
brew install python@3.12

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.12 python3.12-venv


#### 3. Create Virtual Environment
bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate


#### 4. Install Dependencies
bash
# Install all required packages
pip install -r requirements.txt

# Fix numpy compatibility if needed
pip install "numpy>=1.26.0,<2.0" --force-reinstall


#### 5. Run the Application
bash
# Start the Streamlit app

## Features
- *AI vs Real Classification* - Determines if an image is AI-generated or real
- *Confidence Scores* - Shows probability percentages
- *Content Recognition* - Identifies objects in the image
- *GradCAM Visualization* - Shows which areas the model focuses on
- *Detailed Analysis* - Image features and model explanations

## Troubleshooting

### If you get numpy errors:
bash
pip install "numpy>=1.26.0,<2.0" --force-reinstall


### If TensorFlow fails to install:
bash
# Make sure you're using Python 3.12
python --version
# Should show Python 3.12.x


### If the app won't start:
bash
# Make sure virtual environment is activated
source venv/bin/activate
# Then run the app
streamlit run deploy.py


## Model Files
- Ensure AIGeneratedModel.h5 is in the project directory
- The app will warn if model weights can't be loaded

## Usage
1. Upload an image (PNG, JPG, JPEG)
2. Click "Check" button
3. View results:
   - Classification (Real/AI Generated)
   - Confidence percentage
   - Image content description
   - Visual explanation heatmap