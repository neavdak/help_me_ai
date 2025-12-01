@echo off
echo ============================================
echo Qwen 2.5 14B - Offline Setup (Windows)
echo ============================================
echo.
echo This will download ~10GB of model files.
echo After this setup, you can work COMPLETELY OFFLINE.
echo.

REM Create virtual environment
echo -^> Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
echo -^> Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo -^> Installing Python packages...
pip install -r requirements.txt

REM Pre-download the model
echo -^> Pre-downloading Qwen 2.5 14B model (~10GB)...
echo    This will take several minutes...
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; import torch; print('Downloading model and tokenizer...'); model_id = 'Qwen/Qwen2.5-14B-Instruct'; print('-> Downloading tokenizer...'); tokenizer = AutoTokenizer.from_pretrained(model_id); print('-> Downloading model weights...'); model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cpu'); print('✓ Model downloaded and cached successfully!')"

echo.
echo ============================================
echo ✓ Setup Complete!
echo ============================================
echo.
echo You can now work OFFLINE.
echo.
echo To run the app:
echo   .venv\Scripts\activate.bat
echo   streamlit run app.py
echo.
pause
