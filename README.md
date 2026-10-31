
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