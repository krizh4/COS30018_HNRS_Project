
# COS30018_Intelligent-Systems_HNRS_Project
Handwritten Number Recognision Systems


## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# With an NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Without a GPU:
pip install torch torchvision
pip install -r requirements.txt
```

## Run the app

```
.venv\Scripts\Activate.ps1
python web_app.py
```

This opens http://127.0.0.1:5000 in your browser. You need at least one trained model in `checkpoints/`. You can train one in the Train tab. Training rewrites `results/<model>_log.csv`, so don't commit those files unless you mean to.