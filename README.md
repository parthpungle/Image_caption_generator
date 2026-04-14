# 📷 Image Caption Generator (Offline)

A simple AI-powered image caption generator that runs **100% offline** using the BLIP model from Hugging Face. No API keys, no token costs.

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. Open http://localhost:8501 in your browser.

## How It Works

- Uses **BLIP** (Bootstrapping Language-Image Pre-training) from Salesforce
- The model downloads automatically on first run (~1 GB, one-time only)
- After that, everything runs locally — no internet needed
- Optional guided prompts let you steer the caption (e.g., "a photo of", "this image shows")

## Requirements

- Python 3.9+
- ~1 GB disk space for the model
- No GPU required (runs on CPU)
# Image_caption_generator
