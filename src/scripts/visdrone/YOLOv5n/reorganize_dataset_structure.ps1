# Dataset Structure Reorganization Script
# Converts from current structure to standard YOLO format

# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Configuration
$DATASET_PATH = "..\..\..\..\data\my_dataset\visdrone"
$BACKUP_PATH = "..\..\..\..\data\my_dataset\visdrone_backup"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Test-DatasetStructure {
    Write-ColorOutput "[INFO] Analyzing current dataset structure..." "Cyan"
    
    $currentStructure = @{
        "images_train" = Test-Path "$DATASET_PATH\images\train"
        "images_test" = Test-Path "$DATASET_PATH\images\test"
        "images_val" = Test-Path "$DATASET_PATH\images\val"
        "labels_train" = Test-Path "$DATASET_PATH\labels\train"
        "labels_test" = Test-Path "$DATASET_PATH\labels\test"
        "labels_val" = Test-Path "$DATASET_PATH\labels\val"
    }
    
    Write-ColorOutput "[STRUCTURE] Current Structure:" "White"
    foreach ($key in $currentStructure.Keys) {
        $status = if ($currentStructure[$key]) { "[FOUND]" } else { "[NOT FOUND]" }
        Write-ColorOutput "   $key`: $status" "White"
    }
    
    return $currentStructure
}

function Backup-Dataset {
    Write-ColorOutput "[BACKUP] Creating backup..." "Yellow"
    
    if (Test-Path $BACKUP_PATH) {
        Write-ColorOutput "[WARNING] Backup already exists. Removing old backup..." "Yellow"
        Remove-Item -Path $BACKUP_PATH -Recurse -Force
    }
    
    Copy-Item -Path $DATASET_PATH -Destination $BACKUP_PATH -Recurse
    Write-ColorOutput "[SUCCESS] Backup created at: $BACKUP_PATH" "Green"
}

function Reorganize-Dataset {
    param($structure)
    
    Write-ColorOutput "[PROCESS] Reorganizing dataset structure..." "Cyan"
    
    # Create temporary directory for reorganization
    $tempPath = "$DATASET_PATH`_temp"
    New-Item -ItemType Directory -Path $tempPath -Force | Out-Null
    
    # Process each split
    $splits = @("train", "test", "val")
    
    foreach ($split in $splits) {
        if ($structure["images_$split"] -or $structure["labels_$split"]) {
            Write-ColorOutput "[PROCESSING] Processing $split split..." "White"
            
            # Create new structure directories
            $newSplitPath = "$tempPath\$split"
            New-Item -ItemType Directory -Path $newSplitPath -Force | Out-Null
            New-Item -ItemType Directory -Path "$newSplitPath\images" -Force | Out-Null
            New-Item -ItemType Directory -Path "$newSplitPath\labels" -Force | Out-Null
            
            # Move images
            if ($structure["images_$split"]) {
                $sourceImages = "$DATASET_PATH\images\$split"
                if (Test-Path $sourceImages) {
                    $imageFiles = Get-ChildItem -Path $sourceImages -File
                    foreach ($file in $imageFiles) {
                        Move-Item -Path $file.FullName -Destination "$newSplitPath\images\"
                    }
                    Write-ColorOutput "   [SUCCESS] Moved $($imageFiles.Count) images" "Green"
                }
            }
            
            # Move labels
            if ($structure["labels_$split"]) {
                $sourceLabels = "$DATASET_PATH\labels\$split"
                if (Test-Path $sourceLabels) {
                    $labelFiles = Get-ChildItem -Path $sourceLabels -File
                    foreach ($file in $labelFiles) {
                        Move-Item -Path $file.FullName -Destination "$newSplitPath\labels\"
                    }
                    Write-ColorOutput "   [SUCCESS] Moved $($labelFiles.Count) labels" "Green"
                }
            }
        }
    }
    
    # Remove old structure
    Write-ColorOutput "[CLEANUP] Removing old structure..." "Yellow"
    if (Test-Path "$DATASET_PATH\images") {
        Remove-Item -Path "$DATASET_PATH\images" -Recurse -Force
    }
    if (Test-Path "$DATASET_PATH\labels") {
        Remove-Item -Path "$DATASET_PATH\labels" -Recurse -Force
    }
    
    # Move new structure to dataset path
    $tempContent = Get-ChildItem -Path $tempPath
    foreach ($item in $tempContent) {
        Move-Item -Path $item.FullName -Destination $DATASET_PATH
    }
    
    # Clean up temporary directory
    Remove-Item -Path $tempPath -Recurse -Force
    
    Write-ColorOutput "[SUCCESS] Dataset reorganization complete!" "Green"
}

function Verify-NewStructure {
    Write-ColorOutput "[VERIFY] Verifying new structure..." "Cyan"
    
    $splits = @("train", "test", "val")
    $totalImages = 0
    $totalLabels = 0
    
    foreach ($split in $splits) {
        $splitPath = "$DATASET_PATH\$split"
        if (Test-Path $splitPath) {
            $imagesPath = "$splitPath\images"
            $labelsPath = "$splitPath\labels"
            
            $imageCount = 0
            $labelCount = 0
            
            if (Test-Path $imagesPath) {
                $imageCount = (Get-ChildItem -Path $imagesPath -File).Count
                $totalImages += $imageCount
            }
            
            if (Test-Path $labelsPath) {
                $labelCount = (Get-ChildItem -Path $labelsPath -File).Count
                $totalLabels += $labelCount
            }
            
            Write-ColorOutput "   [STATS] $split`: $imageCount images, $labelCount labels" "White"
        }
    }
    
    Write-ColorOutput "[TOTAL] Total: $totalImages images, $totalLabels labels" "Green"
    
    if ($totalImages -gt 0) {
        Write-ColorOutput "[SUCCESS] New structure is valid and ready for augmentation!" "Green"
        return $true
    } else {
        Write-ColorOutput "[ERROR] No images found in new structure!" "Red"
        return $false
    }
}

function Show-NextSteps {
    Write-ColorOutput "[NEXT] Next Steps:" "Cyan"
    Write-ColorOutput "   1. Run the augmentation script: .\run_phase1_augmentation.ps1" "White"
    Write-ColorOutput "   2. The dataset is now in standard YOLO format" "White"
    Write-ColorOutput "   3. Backup is available at: $BACKUP_PATH" "White"
    Write-ColorOutput "   4. If issues occur, you can restore from backup" "White"
}

function Main {
    Write-ColorOutput "[START] Dataset Structure Reorganization for YOLOv5n + VisDrone" "Magenta"
    Write-ColorOutput "[INFO] Converting to Standard YOLO Format" "Magenta"
    Write-ColorOutput ""
    
    try {
        # Step 1: Analyze current structure
        $structure = Test-DatasetStructure
        
        # Check if reorganization is needed
        $needsReorganization = $structure["images_train"] -or $structure["images_test"] -or $structure["images_val"]
        
        if (-not $needsReorganization) {
            Write-ColorOutput "[SUCCESS] Dataset is already in standard YOLO format!" "Green"
            return
        }
        
        # Step 2: Create backup
        Backup-Dataset
        
        # Step 3: Reorganize dataset
        Reorganize-Dataset $structure
        
        # Step 4: Verify new structure
        $success = Verify-NewStructure
        
        if ($success) {
            # Step 5: Show next steps
            Show-NextSteps
            
            Write-ColorOutput "[COMPLETE] Dataset reorganization completed successfully!" "Green"
        } else {
            Write-ColorOutput "[ERROR] Reorganization failed. Check backup at: $BACKUP_PATH" "Red"
        }
        
    } catch {
        Write-ColorOutput "[ERROR] Error during reorganization: $($_.Exception.Message)" "Red"
        Write-ColorOutput "[BACKUP] Backup is available at: $BACKUP_PATH" "Yellow"
    }
}

# Execute main function
Main 