# Filename: setup_yolov5_env.ps1
# Purpose: Set up Python 3.10 virtual environment for YOLOv5 training

Write-Host "🔧 Setting up YOLOv5 environment..."

# Step 1: Check for Python 3.10
$pythonPath = (Get-Command py).Source
$pythonAvailable = & py -3.10 --version 2>$null

if (-not $?) {
    Write-Host "❌ Python 3.10 not found. Please install it from https://www.python.org/downloads/release/python-3100/"
    exit
}

# Step 2: Create virtual environment
Write-Host "📦 Creating virtual environment: yolov5-env"
& py -3.10 -m venv yolov5-env

# Step 3: Activate environment
Write-Host "🚀 Activating environment..."
& .\yolov5-env\Scripts\Activate.ps1

# Step 4: Upgrade pip
pip install --upgrade pip

# Step 5: Install PyTorch (CPU-only for compatibility; change to cu117 for GPU)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 6: Install YOLOv5 requirements
pip install -r src\models\YOLOv5\requirements.txt

# Step 7: Verify installation
python -c "import torch; import torchvision; from torchvision import transforms; print('✅ PyTorch and torchvision installed successfully')"

Write-Host "🎉 YOLOv5 environment setup complete!"
