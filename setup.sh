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

# Pre-download the models
echo "→ Pre-downloading Qwen 2.5 models..."
echo "   This will download BOTH 7B (~8GB) and 14B (~15GB) models."
echo "   Total: ~23GB. This will take some time..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download 7B model
print('\n📦 Downloading 7B model (recommended for 8GB VRAM)...')
model_7b = 'Qwen/Qwen2.5-7B-Instruct'
print('→ Downloading 7B tokenizer...')
tokenizer_7b = AutoTokenizer.from_pretrained(model_7b)
print('→ Downloading 7B model weights...')
model_7b_loaded = AutoModelForCausalLM.from_pretrained(
    model_7b,
    torch_dtype=torch.float16,
    device_map='cpu'
)
print('✓ 7B model downloaded!')

# Download 14B model
print('\n📦 Downloading 14B model (higher quality)...')
model_14b = 'Qwen/Qwen2.5-14B-Instruct'
print('→ Downloading 14B tokenizer...')
tokenizer_14b = AutoTokenizer.from_pretrained(model_14b)
print('→ Downloading 14B model weights...')
model_14b_loaded = AutoModelForCausalLM.from_pretrained(
    model_14b,
    torch_dtype=torch.float16,
    device_map='cpu'
)
print('✓ 14B model downloaded!')

print('\n✓ Both models downloaded and cached successfully!')
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
