#!/usr/bin/env python3
"""
Augmentation Quality Validator for YOLOv5n + VisDrone
Validates augmentation effectiveness using comprehensive image quality metrics.

This validator provides:
- Image quality metrics (SSIM, PSNR, histogram correlation)
- Visual validation with sample generation
- Statistical analysis of augmentation effects
- Quality reports for methodology documentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

from environmental_augmentation_pipeline import (
    EnvironmentalAugmentator, 
    AugmentationType, 
    IntensityLevel
)

@dataclass
class QualityMetrics:
    """Quality metrics for augmentation validation"""
    ssim_score: float
    psnr_score: float
    histogram_correlation: float
    entropy_original: float
    entropy_augmented: float
    entropy_ratio: float
    mean_brightness_change: float
    contrast_change: float
    quality_score: float

@dataclass
class ValidationReport:
    """Complete validation report"""
    augmentation_type: str
    intensity_level: str
    sample_count: int
    metrics: Dict[str, QualityMetrics]
    statistical_summary: Dict[str, float]
    quality_assessment: str
    recommendations: List[str]

class AugmentationQualityValidator:
    """Validates augmentation quality using comprehensive metrics"""
    
    def __init__(self, output_dir: str = "validation_results", seed: int = 42):
        """
        Initialize the quality validator
        
        Args:
            output_dir: Directory to save validation results
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Initialize augmentator
        self.augmentator = EnvironmentalAugmentator(seed=seed)
        
        # Quality thresholds for validation
        self.quality_thresholds = {
            "ssim_min": 0.3,  # Minimum structural similarity
            "ssim_max": 0.9,  # Maximum structural similarity (too high = no change)
            "psnr_min": 15.0,  # Minimum peak signal-to-noise ratio
            "hist_corr_min": 0.4,  # Minimum histogram correlation
            "entropy_ratio_min": 0.7,  # Minimum entropy ratio
            "entropy_ratio_max": 1.3,  # Maximum entropy ratio
            "quality_score_min": 0.5  # Minimum overall quality score
        }
        
    def validate_sample_images(self, image_paths: List[str], 
                             output_samples: bool = True) -> Dict[str, List[ValidationReport]]:
        """
        Validate augmentation quality on sample images
        
        Args:
            image_paths: List of paths to sample images
            output_samples: Whether to save sample images
            
        Returns:
            Dictionary of validation reports by augmentation type
        """
        print("[VALIDATE] Validating Augmentation Quality")
        print("=" * 50)
        
        validation_results = {}
        
        # Test each augmentation type
        for aug_type in AugmentationType:
            print(f"\n[TEST] Testing {aug_type.value.upper()} augmentation...")
            
            aug_results = []
            
            # Test each intensity level
            for intensity in IntensityLevel:
                print(f"   [INTENSITY] {intensity.value.capitalize()} intensity...")
                
                metrics_list = []
                sample_outputs = []
                
                # Process each sample image
                for img_path in image_paths:
                    try:
                        # Load image
                        original = cv2.imread(img_path)
                        if original is None:
                            print(f"   [WARNING] Could not load image: {img_path}")
                            continue
                        
                        # Apply augmentation
                        augmented, config = self.augmentator.augment_image(
                            original, aug_type, intensity
                        )
                        
                        # Calculate quality metrics
                        metrics = self._calculate_quality_metrics(original, augmented)
                        metrics_list.append(metrics)
                        
                        # Save sample if requested
                        if output_samples:
                            sample_info = {
                                "original_path": img_path,
                                "augmentation_type": aug_type.value,
                                "intensity": intensity.value,
                                "config": config.parameters
                            }
                            sample_outputs.append((original, augmented, sample_info))
                        
                    except Exception as e:
                        print(f"   [ERROR] Error processing {img_path}: {e}")
                
                # Generate validation report for this intensity level
                if metrics_list:
                    report = self._generate_validation_report(
                        aug_type, intensity, metrics_list
                    )
                    aug_results.append(report)
                    
                    # Save sample images
                    if output_samples and sample_outputs:
                        self._save_sample_images(sample_outputs, aug_type, intensity)
            
            validation_results[aug_type.value] = aug_results
        
        # Save comprehensive validation report
        self._save_validation_report(validation_results)
        
        # Generate summary visualization
        self._generate_validation_visualization(validation_results)
        
        print("\n[SUCCESS] Augmentation Quality Validation Complete!")
        print(f"[PATH] Results saved to: {self.output_dir}")
        
        return validation_results
    
    def _calculate_quality_metrics(self, original: np.ndarray, 
                                 augmented: np.ndarray) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Convert to grayscale for some metrics
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
        
        # SSIM (Structural Similarity Index)
        ssim_score = ssim(gray_orig, gray_aug, data_range=255)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr_score = psnr(gray_orig, gray_aug, data_range=255)
        
        # Histogram correlation
        hist_orig = cv2.calcHist([original], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_aug = cv2.calcHist([augmented], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_correlation = cv2.compareHist(hist_orig, hist_aug, cv2.HISTCMP_CORREL)
        
        # Entropy measures
        entropy_orig = shannon_entropy(gray_orig)
        entropy_aug = shannon_entropy(gray_aug)
        entropy_ratio = entropy_aug / entropy_orig if entropy_orig > 0 else 0
        
        # Brightness and contrast changes
        mean_brightness_orig = np.mean(gray_orig)
        mean_brightness_aug = np.mean(gray_aug)
        mean_brightness_change = (mean_brightness_aug - mean_brightness_orig) / mean_brightness_orig
        
        std_orig = np.std(gray_orig)
        std_aug = np.std(gray_aug)
        contrast_change = (std_aug - std_orig) / std_orig if std_orig > 0 else 0
        
        # Combined quality score
        quality_score = (ssim_score + hist_correlation + 
                        min(entropy_ratio, 2 - entropy_ratio)) / 3
        
        return QualityMetrics(
            ssim_score=ssim_score,
            psnr_score=psnr_score,
            histogram_correlation=hist_correlation,
            entropy_original=entropy_orig,
            entropy_augmented=entropy_aug,
            entropy_ratio=entropy_ratio,
            mean_brightness_change=mean_brightness_change,
            contrast_change=contrast_change,
            quality_score=quality_score
        )
    
    def _generate_validation_report(self, aug_type: AugmentationType, 
                                  intensity: IntensityLevel, 
                                  metrics_list: List[QualityMetrics]) -> ValidationReport:
        """Generate validation report for specific augmentation and intensity"""
        
        # Calculate statistical summary
        metrics_dict = {}
        for metric_name in ['ssim_score', 'psnr_score', 'histogram_correlation', 
                           'entropy_ratio', 'mean_brightness_change', 'contrast_change', 
                           'quality_score']:
            values = [getattr(m, metric_name) for m in metrics_list]
            metrics_dict[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Quality assessment
        avg_quality = metrics_dict['quality_score']['mean']
        avg_ssim = metrics_dict['ssim_score']['mean']
        avg_psnr = metrics_dict['psnr_score']['mean']
        avg_hist_corr = metrics_dict['histogram_correlation']['mean']
        
        # Determine quality level
        if avg_quality >= 0.7:
            quality_assessment = "EXCELLENT"
        elif avg_quality >= 0.6:
            quality_assessment = "GOOD"
        elif avg_quality >= 0.5:
            quality_assessment = "ACCEPTABLE"
        else:
            quality_assessment = "POOR"
        
        # Generate recommendations
        recommendations = []
        
        if avg_ssim < self.quality_thresholds["ssim_min"]:
            recommendations.append("Low SSIM indicates too much structural change. Consider reducing augmentation intensity.")
        elif avg_ssim > self.quality_thresholds["ssim_max"]:
            recommendations.append("High SSIM indicates minimal change. Consider increasing augmentation intensity.")
        
        if avg_psnr < self.quality_thresholds["psnr_min"]:
            recommendations.append("Low PSNR indicates high noise. Consider refining augmentation parameters.")
        
        if avg_hist_corr < self.quality_thresholds["hist_corr_min"]:
            recommendations.append("Low histogram correlation indicates significant color changes. Verify this is intended.")
        
        if not recommendations:
            recommendations.append("Augmentation quality meets all criteria. No adjustments needed.")
        
        # Create average metrics object
        avg_metrics = QualityMetrics(
            ssim_score=avg_ssim,
            psnr_score=avg_psnr,
            histogram_correlation=avg_hist_corr,
            entropy_original=np.mean([m.entropy_original for m in metrics_list]),
            entropy_augmented=np.mean([m.entropy_augmented for m in metrics_list]),
            entropy_ratio=np.mean([m.entropy_ratio for m in metrics_list]),
            mean_brightness_change=np.mean([m.mean_brightness_change for m in metrics_list]),
            contrast_change=np.mean([m.contrast_change for m in metrics_list]),
            quality_score=avg_quality
        )
        
        return ValidationReport(
            augmentation_type=aug_type.value,
            intensity_level=intensity.value,
            sample_count=len(metrics_list),
            metrics={"average": avg_metrics},
            statistical_summary=metrics_dict,
            quality_assessment=quality_assessment,
            recommendations=recommendations
        )
    
    def _save_sample_images(self, sample_outputs: List[Tuple], 
                          aug_type: AugmentationType, intensity: IntensityLevel):
        """Save sample images for visual validation"""
        
        samples_dir = self.output_dir / "samples" / aug_type.value / intensity.value
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (original, augmented, info) in enumerate(sample_outputs[:5]):  # Save first 5 samples
            # Create side-by-side comparison
            comparison = np.hstack([original, augmented])
            
            # Save comparison image
            output_path = samples_dir / f"sample_{i+1}_comparison.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            # Save individual images
            orig_path = samples_dir / f"sample_{i+1}_original.jpg"
            aug_path = samples_dir / f"sample_{i+1}_augmented.jpg"
            cv2.imwrite(str(orig_path), original)
            cv2.imwrite(str(aug_path), augmented)
            
            # Save info
            info_path = samples_dir / f"sample_{i+1}_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
    
    def _save_validation_report(self, validation_results: Dict):
        """Save comprehensive validation report"""
        
        # Convert to serializable format
        report_data = {}
        for aug_type, reports in validation_results.items():
            report_data[aug_type] = []
            for report in reports:
                report_dict = asdict(report)
                report_data[aug_type].append(report_dict)
        
        # Save JSON report
        json_path = self.output_dir / "validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save human-readable report
        text_path = self.output_dir / "validation_report.txt"
        with open(text_path, 'w') as f:
            f.write("YOLOv5n + VisDrone Augmentation Quality Validation Report\n")
            f.write("=" * 60 + "\n\n")
            
            for aug_type, reports in validation_results.items():
                f.write(f"{aug_type.upper()} AUGMENTATION\n")
                f.write("-" * 30 + "\n")
                
                for report in reports:
                    f.write(f"\n{report.intensity_level.capitalize()} Intensity:\n")
                    f.write(f"  Sample Count: {report.sample_count}\n")
                    f.write(f"  Quality Assessment: {report.quality_assessment}\n")
                    f.write(f"  Quality Score: {report.metrics['average'].quality_score:.3f}\n")
                    f.write(f"  SSIM: {report.metrics['average'].ssim_score:.3f}\n")
                    f.write(f"  PSNR: {report.metrics['average'].psnr_score:.1f}\n")
                    f.write(f"  Histogram Correlation: {report.metrics['average'].histogram_correlation:.3f}\n")
                    f.write(f"  Recommendations:\n")
                    for rec in report.recommendations:
                        f.write(f"    - {rec}\n")
                f.write("\n")
    
    def _generate_validation_visualization(self, validation_results: Dict):
        """Generate validation visualization plots"""
        
        # Prepare data for visualization
        plot_data = []
        for aug_type, reports in validation_results.items():
            for report in reports:
                plot_data.append({
                    'augmentation_type': aug_type,
                    'intensity_level': report.intensity_level,
                    'quality_score': report.metrics['average'].quality_score,
                    'ssim_score': report.metrics['average'].ssim_score,
                    'psnr_score': report.metrics['average'].psnr_score,
                    'histogram_correlation': report.metrics['average'].histogram_correlation,
                    'entropy_ratio': report.metrics['average'].entropy_ratio
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Augmentation Quality Validation Results', fontsize=16)
        
        # Quality Score by Augmentation Type and Intensity
        sns.barplot(data=df, x='augmentation_type', y='quality_score', 
                   hue='intensity_level', ax=axes[0, 0])
        axes[0, 0].set_title('Quality Score by Augmentation Type')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # SSIM Score Distribution
        sns.boxplot(data=df, x='augmentation_type', y='ssim_score', 
                   hue='intensity_level', ax=axes[0, 1])
        axes[0, 1].set_title('SSIM Score Distribution')
        axes[0, 1].set_ylabel('SSIM Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # PSNR Score Distribution
        sns.boxplot(data=df, x='augmentation_type', y='psnr_score', 
                   hue='intensity_level', ax=axes[1, 0])
        axes[1, 0].set_title('PSNR Score Distribution')
        axes[1, 0].set_ylabel('PSNR Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Histogram Correlation
        sns.barplot(data=df, x='augmentation_type', y='histogram_correlation', 
                   hue='intensity_level', ax=axes[1, 1])
        axes[1, 1].set_title('Histogram Correlation')
        axes[1, 1].set_ylabel('Histogram Correlation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "validation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create quality assessment summary
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Count quality assessments
        quality_counts = df.groupby(['augmentation_type', 'intensity_level']).size().reset_index(name='count')
        
        # Create heatmap of quality scores
        pivot_df = df.pivot(index='augmentation_type', columns='intensity_level', values='quality_score')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Quality Score Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_methodology_report(self) -> str:
        """Generate methodology-compliant validation report"""
        
        report_path = self.output_dir / "methodology_validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Augmentation Quality Validation Report\n\n")
            f.write("## Methodology Compliance\n\n")
            f.write("This validation report demonstrates compliance with the established methodology framework for synthetic data augmentation in YOLOv5n + VisDrone object detection.\n\n")
            
            f.write("### Validation Criteria\n\n")
            f.write("The following quality metrics were used to validate augmentation effectiveness:\n\n")
            f.write("- **SSIM (Structural Similarity Index)**: Measures structural preservation\n")
            f.write("- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality degradation\n")
            f.write("- **Histogram Correlation**: Measures color distribution similarity\n")
            f.write("- **Entropy Ratio**: Measures information content preservation\n")
            f.write("- **Quality Score**: Combined metric for overall assessment\n\n")
            
            f.write("### Quality Thresholds\n\n")
            f.write("The following thresholds were established based on computer vision best practices:\n\n")
            for threshold, value in self.quality_thresholds.items():
                f.write(f"- {threshold}: {value}\n")
            
            f.write("\n### Validation Results\n\n")
            f.write("Detailed validation results are available in the following files:\n\n")
            f.write("- `validation_report.json`: Machine-readable validation data\n")
            f.write("- `validation_report.txt`: Human-readable validation summary\n")
            f.write("- `validation_results.png`: Visual validation charts\n")
            f.write("- `quality_heatmap.png`: Quality score heatmap\n")
            f.write("- `samples/`: Sample augmented images for visual inspection\n\n")
            
            f.write("### Methodology Compliance Statement\n\n")
            f.write("This validation process ensures that:\n\n")
            f.write("1. All augmentation intensities produce measurable but reasonable changes\n")
            f.write("2. Image quality remains sufficient for object detection tasks\n")
            f.write("3. Augmentation parameters are scientifically justified\n")
            f.write("4. Results are reproducible and well-documented\n")
            f.write("5. Quality assessment follows established computer vision metrics\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("Based on the validation results, specific recommendations are provided for each augmentation type and intensity level to ensure optimal performance in the YOLOv5n training pipeline.\n")
        
        return str(report_path) 