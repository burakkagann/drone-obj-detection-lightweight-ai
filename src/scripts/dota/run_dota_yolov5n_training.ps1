# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to clean up cache files
function Clean-Cache {
    Write-Host "🧹 Cleaning up DOTA cache files..."
    Remove-Item -Recurse -Force "data\my_dataset\dota\dota-v1.0\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "data\my_dataset\dota\dota-v1.0\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 2: Function to train YOLOv5 on DOTA dataset
function Train-YOLOv5 {
    Write-Host "🚀 Starting YOLOv5 training on DOTA..."
    python src\models\YOLOv5\train.py `
        --img 640 `
        --batch 16 `
        --epochs 50 `
        --data config\dota.yaml `
        --weights yolov5s.pt `
        --name yolo_dota_baseline
    return $LASTEXITCODE
}

# Main script execution
function Run-Training {
    Clean-Cache
    $code = Train-YOLOv5
    if ($code -eq 0) {
        Write-Host "✅ YOLOv5 training on DOTA dataset completed successfully!"
    } else {
        Write-Host "❌ YOLOv5 training failed with exit code $code"
        exit $code
    }
}

Run-Training
