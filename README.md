# AI Documentation Generator & Chat Assistant

This is a local AI application powered by **Qwen 2.5 (14B)**. It allows you to:
1.  **Chat** with the model (with file context support).
2.  **Generate Documentation** automatically from code changes (Git diffs) and templates.

It is designed to run offline on consumer hardware (e.g., NVIDIA RTX 4060 with 8GB VRAM).

## Features
- **Offline & Private**: Runs entirely on your local machine.
- **Large Context**: Supports up to 32k-128k tokens context (depending on VRAM).
- **Doc Generator**: Analyzes your code changes and fills out documentation templates (PDF/Word).
- **File Support**: Upload PDF, DOCX, TXT, MD, PY files for context.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

2.  **First Run**: The application will automatically download the `Qwen/Qwen2.5-14B-Instruct` model (~10GB) from Hugging Face.

3.  **Modes**:
    - **Chat**: Upload files and ask questions.
    - **Doc Generator**: Point to a code folder, upload a template, and generate reports.

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060 or better).
- **RAM**: 32GB+ recommended (64GB is ideal for layer offloading).
- **OS**: Linux or Windows (with WSL2).
- **Note**: With 8GB VRAM, some layers will offload to RAM. This is automatic and works seamlessly.
