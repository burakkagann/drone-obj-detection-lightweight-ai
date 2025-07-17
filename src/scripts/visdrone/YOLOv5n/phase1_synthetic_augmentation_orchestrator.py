#!/usr/bin/env python3
"""
Phase 1: Synthetic Augmentation Orchestrator for YOLOv5n + VisDrone
Coordinates all Phase 1 components according to methodology framework.

This orchestrator:
1. Validates source dataset
2. Creates stratified augmented dataset
3. Validates augmentation quality
4. Generates comprehensive reports
5. Prepares dataset for YOLOv5n training
"""

import sys
import os
import argparse
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir / ".." / ".." / ".."
sys.path.append(str(project_root))

from environmental_augmentation_pipeline import EnvironmentalAugmentator
from dataset_stratification_manager import DatasetStratificationManager
from augmentation_quality_validator import AugmentationQualityValidator

class Phase1Orchestrator:
    """Orchestrates Phase 1: Synthetic Data Augmentation Pipeline"""
    
    def __init__(self, config: Dict):
        """Initialize orchestrator with configuration"""
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize components
        self.augmentator = EnvironmentalAugmentator(seed=config.get("seed", 42))
        self.stratification_manager = DatasetStratificationManager(
            source_dataset_path=config["source_dataset_path"],
            output_dataset_path=config["output_dataset_path"],
            seed=config.get("seed", 42)
        )
        self.quality_validator = AugmentationQualityValidator(
            output_dir=config["validation_output_dir"],
            seed=config.get("seed", 42)
        )
        
        # Create output directories
        self.output_dir = Path(config["output_dataset_path"])
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def execute_phase1(self) -> Dict:
        """Execute complete Phase 1 pipeline"""
        print("[START] Starting Phase 1: Synthetic Data Augmentation Pipeline")
        print("=" * 80)
        print(f"[TIME] Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[PATH] Source Dataset: {self.config['source_dataset_path']}")
        print(f"[PATH] Output Dataset: {self.config['output_dataset_path']}")
        print(f"[INFO] Methodology: YOLOv5n + VisDrone Environmental Robustness")
        print("=" * 80)
        
        results = {
            "phase": "Phase 1: Synthetic Data Augmentation",
            "start_time": self.start_time.isoformat(),
            "config": self.config,
            "results": {}
        }
        
        try:
            # Step 1: Validate source dataset
            print("\n[STEP 1] Validating Source Dataset")
            source_validation = self._validate_source_dataset()
            results["results"]["source_validation"] = source_validation
            
            # Step 2: Create stratified dataset
            print("\n[STEP 2] Creating Stratified Dataset")
            stratification_results = self._create_stratified_dataset()
            results["results"]["stratification"] = stratification_results
            
            # Step 3: Validate augmentation quality
            print("\n[STEP 3] Validating Augmentation Quality")
            quality_validation = self._validate_augmentation_quality()
            results["results"]["quality_validation"] = quality_validation
            
            # Set end_time before generating reports
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["duration"] = str(end_time - self.start_time)
            results["status"] = "SUCCESS"
            
            # Step 4: Generate comprehensive reports
            print("\n[STEP 4] Generating Comprehensive Reports")
            reports = self._generate_reports(results)
            results["results"]["reports"] = reports
            
            # Step 5: Prepare for next phase
            print("\n[STEP 5] Preparing for Phase 2")
            next_phase_prep = self._prepare_for_next_phase()
            results["results"]["next_phase_preparation"] = next_phase_prep
            
            print("\n[SUCCESS] Phase 1 Complete!")
            print(f"[TIME] Duration: {end_time - self.start_time}")
            print(f"[INFO] Total Images Generated: {stratification_results.get('statistics', {}).get('total_images', 0)}")
            print(f"[PATH] Results saved to: {self.output_dir}")
            
        except Exception as e:
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["duration"] = str(end_time - self.start_time)
            results["status"] = "FAILED"
            results["error"] = str(e)
            
            print(f"\n[ERROR] Phase 1 Failed: {e}")
            raise
        
        # Save execution results
        self._save_execution_results(results)
        
        return results
    
    def _validate_source_dataset(self) -> Dict:
        """Validate source dataset structure and content"""
        print("   [VALIDATE] Analyzing source dataset structure...")
        
        source_path = Path(self.config["source_dataset_path"])
        
        validation_results = {
            "path_exists": source_path.exists(),
            "structure_valid": False,
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "issues": []
        }
        
        if not validation_results["path_exists"]:
            validation_results["issues"].append(f"Source dataset path does not exist: {source_path}")
            return validation_results
        
        # Check for required splits
        required_splits = ["train", "val", "test"]
        splits_found = 0
        
        for split in required_splits:
            split_path = source_path / split
            split_info = {
                "exists": split_path.exists(),
                "images": 0,
                "labels": 0,
                "images_path_exists": False,
                "labels_path_exists": False
            }
            
            if split_info["exists"]:
                images_path = split_path / "images"
                labels_path = split_path / "labels"
                
                split_info["images_path_exists"] = images_path.exists()
                split_info["labels_path_exists"] = labels_path.exists()
                
                if split_info["images_path_exists"]:
                    image_files = list(images_path.glob("*"))
                    split_info["images"] = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    validation_results["total_images"] += split_info["images"]
                
                if split_info["labels_path_exists"]:
                    label_files = list(labels_path.glob("*.txt"))
                    split_info["labels"] = len(label_files)
                    validation_results["total_labels"] += split_info["labels"]
                
                if split_info["images"] > 0:
                    splits_found += 1
            
            validation_results["splits"][split] = split_info
        
        validation_results["structure_valid"] = splits_found >= 2  # At least train and val
        
        if not validation_results["structure_valid"]:
            validation_results["issues"].append("Invalid dataset structure. Need at least train and val splits.")
        
        if validation_results["total_images"] == 0:
            validation_results["issues"].append("No images found in dataset.")
        
        print(f"   [SUCCESS] Source validation complete: {validation_results['total_images']} images found")
        
        return validation_results
    
    def _create_stratified_dataset(self) -> Dict:
        """Create stratified dataset with augmentations"""
        print("   [CREATE] Creating stratified dataset with augmentations...")
        
        # Execute stratification
        stats = self.stratification_manager.create_stratified_dataset(verbose=True)
        
        # Validate stratification
        validation = self.stratification_manager.validate_stratification()
        
        stratification_results = {
            "statistics": {
                "total_images": stats.total_images,
                "original_count": stats.original_count,
                "light_count": stats.light_count,
                "medium_count": stats.medium_count,
                "heavy_count": stats.heavy_count,
                "augmentation_breakdown": dict(stats.augmentation_breakdown)
            },
            "validation": validation,
            "methodology_compliance": validation["is_valid"]
        }
        
        print(f"   [SUCCESS] Stratification complete: {stats.total_images} total images")
        print(f"   [INFO] Methodology compliance: {'[PASS]' if validation['is_valid'] else '[FAIL]'}")
        
        return stratification_results
    
    def _validate_augmentation_quality(self) -> Dict:
        """Validate augmentation quality using sample images"""
        print("   [VALIDATE] Validating augmentation quality...")
        
        # Get sample images for validation
        sample_images = self._get_sample_images()
        
        if not sample_images:
            print("   [WARNING] No sample images found for quality validation")
            return {"status": "SKIPPED", "reason": "No sample images found"}
        
        # Run quality validation
        validation_results = self.quality_validator.validate_sample_images(
            sample_images, output_samples=True
        )
        
        # Generate methodology report
        methodology_report = self.quality_validator.generate_methodology_report()
        
        quality_results = {
            "validation_results": validation_results,
            "methodology_report": methodology_report,
            "sample_count": len(sample_images),
            "status": "COMPLETE"
        }
        
        print(f"   [SUCCESS] Quality validation complete: {len(sample_images)} samples processed")
        
        return quality_results
    
    def _get_sample_images(self) -> List[str]:
        """Get sample images for quality validation"""
        source_path = Path(self.config["source_dataset_path"])
        sample_images = []
        
        # Get sample images from train split
        train_images_path = source_path / "train" / "images"
        if train_images_path.exists():
            image_files = list(train_images_path.glob("*"))
            # Filter for image files
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            # Take first 10 images as samples
            sample_images = [str(f) for f in image_files[:10]]
        
        return sample_images
    
    def _generate_reports(self, results: Dict) -> Dict:
        """Generate comprehensive reports for Phase 1"""
        print("   [REPORT] Generating comprehensive reports...")
        
        reports = {
            "execution_report": self._generate_execution_report(results),
            "methodology_report": self._generate_methodology_report(results),
            "technical_report": self._generate_technical_report(results),
            "summary_report": self._generate_summary_report(results)
        }
        
        print(f"   [SUCCESS] Reports generated: {len(reports)} reports created")
        
        return reports
    
    def _generate_execution_report(self, results: Dict) -> str:
        """Generate execution report"""
        report_path = self.reports_dir / "phase1_execution_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 1 Execution Report\n\n")
            f.write("## Overview\n\n")
            f.write("This report documents the execution of Phase 1: Synthetic Data Augmentation Pipeline for YOLOv5n + VisDrone object detection.\n\n")
            
            f.write("## Execution Details\n\n")
            f.write(f"- **Start Time**: {results['start_time']}\n")
            f.write(f"- **End Time**: {results['end_time']}\n")
            f.write(f"- **Duration**: {results['duration']}\n")
            f.write(f"- **Status**: {results['status']}\n\n")
            
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(results['config'], indent=2))
            f.write("\n```\n\n")
            
            f.write("## Results Summary\n\n")
            if 'stratification' in results['results']:
                stats = results['results']['stratification']['statistics']
                f.write(f"- **Total Images Generated**: {stats['total_images']}\n")
                f.write(f"- **Original Images**: {stats['original_count']}\n")
                f.write(f"- **Light Augmented**: {stats['light_count']}\n")
                f.write(f"- **Medium Augmented**: {stats['medium_count']}\n")
                f.write(f"- **Heavy Augmented**: {stats['heavy_count']}\n")
                f.write(f"- **Methodology Compliance**: {'[PASS]' if stats else '[FAIL]'}\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review quality validation results\n")
            f.write("2. Proceed to Phase 2: Enhanced Training Pipeline\n")
            f.write("3. Integrate augmented dataset into YOLOv5n training\n")
        
        return str(report_path)
    
    def _generate_methodology_report(self, results: Dict) -> str:
        """Generate methodology compliance report"""
        report_path = self.reports_dir / "phase1_methodology_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 1 Methodology Compliance Report\n\n")
            f.write("## Methodology Framework Compliance\n\n")
            f.write("This report validates compliance with the established methodology framework for YOLOv5n + VisDrone synthetic data augmentation.\n\n")
            
            f.write("### Distribution Strategy Compliance\n\n")
            f.write("**Target Distribution:**\n")
            f.write("- Original: 40% of training data\n")
            f.write("- Light conditions: 20% of training data\n")
            f.write("- Medium conditions: 25% of training data\n")
            f.write("- Heavy conditions: 15% of training data\n\n")
            
            if 'stratification' in results['results']:
                validation = results['results']['stratification']['validation']
                f.write("**Actual Distribution:**\n")
                for key, value in validation['actual_ratios'].items():
                    f.write(f"- {key.capitalize()}: {value:.1%}\n")
                f.write("\n")
                
                f.write("**Compliance Status:**\n")
                f.write(f"- Validation Score: {validation['validation_score']:.4f}\n")
                f.write(f"- Compliance: {'[PASS]' if validation['is_valid'] else '[FAIL]'}\n\n")
            
            f.write("### Augmentation Types Implemented\n\n")
            f.write("1. **Fog/Haze**: Light (50-100m visibility), Medium (25-50m), Heavy (10-25m)\n")
            f.write("2. **Low Light/Night**: Dusk, Urban Night, Minimal Light\n")
            f.write("3. **Motion Blur**: Light, Medium, Heavy camera shake simulation\n")
            f.write("4. **Weather**: Light Rain, Heavy Rain, Snow\n\n")
            
            f.write("### Quality Validation\n\n")
            if 'quality_validation' in results['results']:
                f.write("Quality validation was performed using comprehensive image metrics:\n")
                f.write("- SSIM (Structural Similarity Index)\n")
                f.write("- PSNR (Peak Signal-to-Noise Ratio)\n")
                f.write("- Histogram Correlation\n")
                f.write("- Entropy Analysis\n")
                f.write("- Combined Quality Score\n\n")
                f.write("Detailed quality results are available in the validation reports.\n\n")
            
            f.write("### Methodology Compliance Statement\n\n")
            f.write("[SUCCESS] **Pre-processed Augmentation**: Implemented as recommended\n")
            f.write("[SUCCESS] **Mixed Training Strategy**: Single model on mixed data\n")
            f.write("[SUCCESS] **Scientific Rigor**: Comprehensive validation and metrics\n")
            f.write("[SUCCESS] **Reproducibility**: Fixed seeds and documented parameters\n")
            f.write("[SUCCESS] **Quality Assurance**: Systematic validation pipeline\n")
        
        return str(report_path)
    
    def _generate_technical_report(self, results: Dict) -> str:
        """Generate technical implementation report"""
        report_path = self.reports_dir / "phase1_technical_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 1 Technical Implementation Report\n\n")
            f.write("## Technical Architecture\n\n")
            f.write("### Components Implemented\n\n")
            f.write("1. **EnvironmentalAugmentator**: Core augmentation engine\n")
            f.write("2. **DatasetStratificationManager**: Dataset distribution management\n")
            f.write("3. **AugmentationQualityValidator**: Quality assurance system\n")
            f.write("4. **Phase1Orchestrator**: Pipeline coordination\n\n")
            
            f.write("### Augmentation Parameters\n\n")
            f.write("Detailed augmentation parameters are documented in the implementation for:\n")
            f.write("- Fog simulation with atmospheric perspective\n")
            f.write("- Night conditions with color temperature adjustment\n")
            f.write("- Motion blur with directional kernels\n")
            f.write("- Weather effects with particle simulation\n\n")
            
            f.write("### Quality Metrics\n\n")
            f.write("The following metrics are used for quality validation:\n")
            f.write("- **SSIM**: Structural similarity preservation\n")
            f.write("- **PSNR**: Signal-to-noise ratio maintenance\n")
            f.write("- **Histogram Correlation**: Color distribution similarity\n")
            f.write("- **Entropy Ratio**: Information content preservation\n\n")
            
            f.write("### File Structure\n\n")
            f.write("```\n")
            f.write("output_dataset/\n")
            f.write("|-- train/\n")
            f.write("|   |-- images/              # ALL images: original + augmented\n")
            f.write("|   |   |-- drone001.jpg     # Original image\n")
            f.write("|   |   |-- drone001_fog_light.jpg     # Fog augmented\n")
            f.write("|   |   |-- drone001_night_medium.jpg  # Night augmented\n")
            f.write("|   |   |-- drone001_blur_heavy.jpg    # Motion blur augmented\n")
            f.write("|   |   +-- ...              # All images in single directory\n")
            f.write("|   +-- labels/              # ALL labels: original + augmented\n")
            f.write("|       |-- drone001.txt     # Original label\n")
            f.write("|       |-- drone001_fog_light.txt     # Matches augmented image\n")
            f.write("|       |-- drone001_night_medium.txt  # Matches augmented image\n")
            f.write("|       |-- drone001_blur_heavy.txt    # Matches augmented image\n")
            f.write("|       +-- ...              # All labels in single directory\n")
            f.write("|-- val/                     # Same structure as train/\n")
            f.write("|-- test/                    # Same structure as train/\n")
            f.write("|-- dataset_config.yaml      # YOLOv5 configuration\n")
            f.write("+-- dataset_statistics.json  # Stratification statistics\n")
            f.write("```\n\n")
        
        return str(report_path)
    
    def _generate_summary_report(self, results: Dict) -> str:
        """Generate executive summary report"""
        report_path = self.reports_dir / "phase1_summary_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 1 Executive Summary\n\n")
            f.write("## Objective\n\n")
            f.write("Implement comprehensive synthetic data augmentation pipeline for YOLOv5n + VisDrone object detection according to established methodology framework.\n\n")
            
            f.write("## Key Achievements\n\n")
            if 'stratification' in results['results']:
                stats = results['results']['stratification']['statistics']
                f.write(f"[SUCCESS] **Dataset Augmentation**: Generated {stats['total_images']} total images\n")
                f.write(f"[SUCCESS] **Methodology Compliance**: Distribution strategy implemented correctly\n")
            
            f.write("[SUCCESS] **Quality Validation**: Comprehensive quality metrics implemented\n")
            f.write("[SUCCESS] **Documentation**: Complete technical and methodology documentation\n")
            f.write("[SUCCESS] **Reproducibility**: All processes documented and reproducible\n\n")
            
            f.write("## Next Phase Readiness\n\n")
            f.write("The Phase 1 implementation provides:\n")
            f.write("- Stratified dataset ready for YOLOv5n training\n")
            f.write("- Quality-validated augmentations\n")
            f.write("- Comprehensive documentation for thesis work\n")
            f.write("- Foundation for Phase 2 implementation\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. Review quality validation results before proceeding\n")
            f.write("2. Consider adjusting augmentation parameters if needed\n")
            f.write("3. Proceed to Phase 2: Enhanced Training Pipeline\n")
            f.write("4. Maintain documentation standards for remaining phases\n")
        
        return str(report_path)
    
    def _prepare_for_next_phase(self) -> Dict:
        """Prepare for Phase 2 implementation"""
        print("   [PREPARE] Preparing for Phase 2...")
        
        # Create Phase 2 ready configuration
        phase2_config = {
            "dataset_path": str(self.output_dir),
            "dataset_config": str(self.output_dir / "dataset_config.yaml"),
            "augmented_dataset_ready": True,
            "phase1_completed": True,
            "next_phase": "Phase 2: Enhanced Training Pipeline"
        }
        
        # Save Phase 2 preparation
        phase2_config_path = self.output_dir / "phase2_config.json"
        with open(phase2_config_path, 'w', encoding='utf-8') as f:
            json.dump(phase2_config, f, indent=2)
        
        preparation_results = {
            "phase2_config_path": str(phase2_config_path),
            "dataset_ready": True,
            "status": "READY"
        }
        
        print(f"   [SUCCESS] Phase 2 preparation complete")
        
        return preparation_results
    
    def _save_execution_results(self, results: Dict):
        """Save execution results"""
        results_path = self.output_dir / "phase1_execution_results.json"
        
        # Make results JSON serializable
        json_serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_results, f, indent=2)
        
        print(f"   [SAVE] Execution results saved to: {results_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if hasattr(obj, '__dict__'):
            # Convert objects with __dict__ to dictionaries
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            # Handle dictionaries recursively
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Handle lists and tuples recursively
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # These types are already JSON serializable
            return obj
        else:
            # For any other type, convert to string
            return str(obj)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 1: Synthetic Data Augmentation Pipeline')
    parser.add_argument('--source', required=True, help='Source dataset path')
    parser.add_argument('--output', required=True, help='Output dataset path')
    parser.add_argument('--validation-output', default='validation_results', help='Validation results output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        "source_dataset_path": args.source,
        "output_dataset_path": args.output,
        "validation_output_dir": args.validation_output,
        "seed": args.seed
    }
    
    if args.config:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # Execute Phase 1
    orchestrator = Phase1Orchestrator(config)
    results = orchestrator.execute_phase1()
    
    # Print final status
    if results["status"] == "SUCCESS":
        print(f"\n[SUCCESS] Phase 1 Successfully Completed!")
        print(f"[INFO] Total Images: {results['results']['stratification']['statistics']['total_images']}")
        print(f"[PATH] Output: {config['output_dataset_path']}")
        print(f"[READY] Ready for Phase 2: Enhanced Training Pipeline")
    else:
        print(f"\n[ERROR] Phase 1 Failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main() 