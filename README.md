# AI Documentation Generator & Chat Assistant

This is a local **OFFLINE** AI application powered by **Qwen 2.5 (7B)**. It allows you to:
1.  **Chat** with the model (with file context support).
2.  **Generate Documentation** automatically from code changes (Git diffs) and templates.

Works **completely offline** after initial setup.

## Features
- **Fully Offline & Private**: Runs entirely on your local machine after setup.
- **Large Context**: Supports up to 128k tokens context.
- **Doc Generator**: Analyzes your code changes and fills out documentation templates (PDF/Word).
- **File Support**: Upload PDF, DOCX, TXT, MD, PY files for context.

## ONE-TIME Installation (Requires Internet)

**⚠️ IMPORTANT**: You need internet connection ONLY ONCE for setup. After that, everything works offline.

### Quick Setup (Recommended)

**Linux/Mac:**
```bash
git clone https://github.com/neavdak/help_me_ai
cd help_me_ai
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
git clone https://github.com/neavdak/help_me_ai
cd help_me_ai
setup.bat
```

The setup script will:
1. Create a virtual environment
2. Install all Python packages
3.  **First Run**: The application will automatically download the `Qwen/Qwen2.5-7B-Instruct` model (~5GB) from Hugging Face.he
4. Verify everything works

**After setup completes, you can disconnect from the internet permanently.**

### Manual Setup (Alternative)

1.  **Clone & Enter**:
    ```bash
    git clone https://github.com/neavdak/help_me_ai
    cd help_me_ai
    ```

2.  **Create virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pre-download model** (do this while still connected):
    ```bash
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-14B-Instruct', device_map='cpu')"
    ```

## Running Offline

After setup, **disconnect from internet** and run:

```bash
source .venv/bin/activate  # Windows: .venv\Scripts\activate
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Modes

- **Chat Mode**: Upload files and ask questions about them.
- **Doc Generator Mode**: Point to code folder + upload template → get auto-generated documentation.

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060 or better).
- **RAM**: 32GB+ recommended (64GB is ideal for layer offloading).
- **Storage**: 15GB free space (for model + dependencies).
- **OS**: Linux, Windows, or Mac.
- **Note**: With 8GB VRAM, some layers will offload to RAM automatically.
