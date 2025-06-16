# Activate virtual environment
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements/visdrone/requirements_mobilenet-ssd.txt

# Check if CUDA is available
$cuda_available = python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
if ($cuda_available -eq "True") {
    Write-Host "CUDA is available. Using GPU for training."
    $gpu_arg = "0"  # Use first GPU
} else {
    Write-Host "CUDA is not available. Using CPU for training."
    $gpu_arg = "-1"  # Use CPU
}

# Create necessary directories
$directories = @(
    "checkpoints/mobilenet_ssd_visdrone",
    "logs/mobilenet_ssd_visdrone"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

# Start training
Write-Host "Starting MobileNet-SSD training on VisDrone dataset..."
python src/scripts/train_mobilenet_ssd.py --gpu $gpu_arg

# Deactivate virtual environment
deactivate 