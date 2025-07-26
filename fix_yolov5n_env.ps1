# Fix YOLOv5n Environment TensorFlow/TensorBoard Compatibility Issue
# This script resolves the TensorFlow/TensorBoard import error

Write-Host "Fixing YOLOv5n Environment TensorFlow/TensorBoard Compatibility..." -ForegroundColor Green

# Navigate to repository root
Set-Location "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Activate YOLOv5n environment
Write-Host "Activating YOLOv5n environment..." -ForegroundColor Yellow
& .\venvs\yolov5n_env\Scripts\Activate.ps1

# Check current TensorBoard/TensorFlow versions
Write-Host "Current TensorBoard/TensorFlow versions:" -ForegroundColor Yellow
pip list | findstr -i tensor

# Option 1: Install compatible TensorFlow and TensorBoard versions
Write-Host "Installing compatible TensorFlow and TensorBoard versions..." -ForegroundColor Yellow
pip install tensorflow==2.13.0 tensorboard==2.13.0

# Alternative Option 2: Remove TensorFlow if not needed (uncomment if Option 1 fails)
# Write-Host "Removing TensorFlow to eliminate conflicts..." -ForegroundColor Yellow
# pip uninstall tensorflow tensorboard -y
# pip install tensorboard

Write-Host "Environment fix completed!" -ForegroundColor Green
Write-Host "Testing TensorBoard import..." -ForegroundColor Yellow

# Test the fix
python -c "import torch.utils.tensorboard; print('TensorBoard import successful!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: TensorBoard compatibility fixed!" -ForegroundColor Green
} else {
    Write-Host "WARNING: TensorBoard import still failing. Trying alternative fix..." -ForegroundColor Red
    
    # Alternative fix: Remove TensorFlow completely
    pip uninstall tensorflow -y
    pip install tensorboard --no-deps
    
    # Test again
    python -c "import torch.utils.tensorboard; print('TensorBoard import successful!')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Alternative fix worked!" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Manual intervention needed for TensorBoard compatibility" -ForegroundColor Red
    }
}