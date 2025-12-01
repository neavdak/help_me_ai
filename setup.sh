#!/bin/bash
set -e

echo "============================================"
echo "Qwen 2.5 14B - Offline Setup"
echo "============================================"
echo ""
echo "This will download ~10GB of model files."
echo "After this setup, you can work COMPLETELY OFFLINE."
echo ""

# Create virtual environment
echo "→ Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "→ Installing Python packages..."
pip install -r requirements.txt

# Pre-download the model
echo "→ Pre-downloading Qwen 2.5 14B model (~10GB)..."
echo "   This will take several minutes..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Downloading model and tokenizer...')
model_id = 'Qwen/Qwen2.5-14B-Instruct'

# Download tokenizer
print('→ Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Download model (this will cache it locally)
print('→ Downloading model weights...')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='cpu'  # Download only, don't load to GPU yet
)

print('✓ Model downloaded and cached successfully!')
"

echo ""
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "You can now work OFFLINE."
echo ""
echo "To run the app:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo ""
