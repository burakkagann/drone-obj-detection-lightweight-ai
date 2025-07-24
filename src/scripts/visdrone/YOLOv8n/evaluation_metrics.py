#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for YOLOv8n Baseline and Trial-1
Master's Thesis: Robust Object Detection for Surveillance Drones

This module implements comprehensive evaluation metrics according to the thesis methodology:
- Detection Accuracy: mAP, Precision, Recall
- Inference Speed: FPS, inference time (ms)
- Model Size: Memory usage (MB), storage requirements
- Robustness: Performance degradation under synthetic conditions

Author: Burak Kağan Yılmazer
Date: January 2025
Environment: yolov8n-visdrone_venv
"""

import os
import sys
import time
import psutil
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import torch
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    sys.exit(1)

class YOLOv8nEvaluationMetrics:
    """Comprehensive evaluation metrics collector for YOLOv8n models"""
    
    def __init__(self, model_path: str, dataset_config: str, output_dir: Path):
        """
        Initialize evaluation metrics collector
        
        Args:
            model_path: Path to trained YOLOv8n model weights
            dataset_config: Path to dataset configuration YAML
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = YOLO(model_path)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Evaluation results storage
        self.results = {
            'detection_accuracy': {},
            'inference_speed': {},
            'model_size': {},
            'robustness': {},
            'edge_performance': {},
            'metadata': {
                'model_path': str(model_path),
                'dataset_config': str(dataset_config),
                'evaluation_time': datetime.now().isoformat(),
                'hardware': self._get_hardware_info()
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_file = self.output_dir / f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information for evaluation context"""
        hardware_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / 1024**3, 2),
            'python_version': sys.version,
        }
        
        if torch.cuda.is_available():
            hardware_info.update({
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            })
        else:
            hardware_info['gpu_available'] = False
        
        return hardware_info
    
    def evaluate_detection_accuracy(self) -> Dict:
        """
        Evaluate detection accuracy metrics (mAP, Precision, Recall)
        Methodology requirement: Section 4.1 - Detection Accuracy
        """
        self.logger.info("[ACCURACY] Evaluating detection accuracy metrics...")
        
        try:
            # Run validation on dataset
            results = self.model.val(data=self.dataset_config, verbose=True)
            
            # Extract metrics
            accuracy_metrics = {
                'mAP_50': float(results.box.map50),  # mAP@0.5
                'mAP_50_95': float(results.box.map),  # mAP@0.5:0.95
                'precision': float(results.box.mp),   # Mean precision
                'recall': float(results.box.mr),      # Mean recall
                'f1_score': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr)) if (results.box.mp + results.box.mr) > 0 else 0.0
            }
            
            # Per-class metrics if available
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
                class_metrics = {}
                for idx, class_idx in enumerate(results.box.ap_class_index):
                    class_metrics[f'class_{int(class_idx)}_mAP50'] = float(results.box.ap50[idx])
                
                accuracy_metrics['per_class_metrics'] = class_metrics
            
            self.results['detection_accuracy'] = accuracy_metrics
            
            self.logger.info(f"[ACCURACY] mAP@0.5: {accuracy_metrics['mAP_50']:.4f}")
            self.logger.info(f"[ACCURACY] mAP@0.5:0.95: {accuracy_metrics['mAP_50_95']:.4f}")
            self.logger.info(f"[ACCURACY] Precision: {accuracy_metrics['precision']:.4f}")
            self.logger.info(f"[ACCURACY] Recall: {accuracy_metrics['recall']:.4f}")
            self.logger.info(f"[ACCURACY] F1-Score: {accuracy_metrics['f1_score']:.4f}")
            
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"[ERROR] Detection accuracy evaluation failed: {e}")
            return {}
    
    def evaluate_inference_speed(self, num_samples: int = 100, image_size: Tuple[int, int] = (640, 640)) -> Dict:
        """
        Evaluate inference speed metrics (FPS, inference time)
        Methodology requirement: Section 4.1 - Inference Speed
        """
        self.logger.info(f"[SPEED] Evaluating inference speed with {num_samples} samples...")
        
        try:
            # Create dummy images for speed testing
            dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            
            # Warmup runs
            for _ in range(10):
                _ = self.model.predict(dummy_image, verbose=False)
            
            # Time inference
            inference_times = []
            
            start_time = time.time()
            for i in range(num_samples):
                inference_start = time.time()
                _ = self.model.predict(dummy_image, verbose=False)
                inference_end = time.time()
                
                inference_times.append((inference_end - inference_start) * 1000)  # Convert to ms
                
                if (i + 1) % 20 == 0:
                    self.logger.info(f"[SPEED] Completed {i + 1}/{num_samples} inference tests")
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            fps = num_samples / total_time
            
            speed_metrics = {
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'std_inference_time_ms': round(std_inference_time, 2),
                'min_inference_time_ms': round(np.min(inference_times), 2),
                'max_inference_time_ms': round(np.max(inference_times), 2),
                'fps': round(fps, 2),
                'total_samples': num_samples,
                'image_size': image_size
            }
            
            self.results['inference_speed'] = speed_metrics
            
            self.logger.info(f"[SPEED] Average inference time: {avg_inference_time:.2f} ms")
            self.logger.info(f"[SPEED] FPS: {fps:.2f}")
            self.logger.info(f"[SPEED] Std dev: {std_inference_time:.2f} ms")
            
            return speed_metrics
            
        except Exception as e:
            self.logger.error(f"[ERROR] Inference speed evaluation failed: {e}")
            return {}
    
    def evaluate_model_size(self) -> Dict:
        """
        Evaluate model size metrics (memory usage, storage requirements)
        Methodology requirement: Section 4.1 - Model Size
        """
        self.logger.info("[SIZE] Evaluating model size metrics...")
        
        try:
            # Model file size
            model_file_size_mb = os.path.getsize(self.model_path) / (1024 ** 2)
            
            # Memory usage during inference
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 ** 2)  # MB
            
            # Dummy inference to measure memory usage
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_image, verbose=False)
            
            memory_after = process.memory_info().rss / (1024 ** 2)  # MB
            memory_usage_mb = memory_after - memory_before
            
            # Model parameters count
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            size_metrics = {
                'model_file_size_mb': round(model_file_size_mb, 2),
                'memory_usage_mb': round(max(memory_usage_mb, 0), 2),  # Ensure non-negative
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_category': self._categorize_model_size(model_file_size_mb)
            }
            
            self.results['model_size'] = size_metrics
            
            self.logger.info(f"[SIZE] Model file size: {model_file_size_mb:.2f} MB")
            self.logger.info(f"[SIZE] Memory usage: {size_metrics['memory_usage_mb']:.2f} MB")
            self.logger.info(f"[SIZE] Total parameters: {total_params:,}")
            
            return size_metrics
            
        except Exception as e:
            self.logger.error(f"[ERROR] Model size evaluation failed: {e}")
            return {}
    
    def _categorize_model_size(self, size_mb: float) -> str:
        """Categorize model size for edge deployment"""
        if size_mb < 10:
            return "ultra_lightweight"
        elif size_mb < 50:
            return "lightweight"
        elif size_mb < 100:
            return "medium"
        else:
            return "heavy"
    
    def evaluate_robustness_degradation(self, baseline_metrics: Dict) -> Dict:
        """
        Evaluate robustness and performance degradation
        Methodology requirement: Section 4.1 - Robustness, Section 4.2 - Synthetic Data Impact
        """
        self.logger.info("[ROBUSTNESS] Evaluating robustness metrics...")
        
        try:
            current_map50 = self.results.get('detection_accuracy', {}).get('mAP_50', 0)
            baseline_map50 = baseline_metrics.get('detection_accuracy', {}).get('mAP_50', 0)
            
            if baseline_map50 > 0:
                performance_change = ((current_map50 - baseline_map50) / baseline_map50) * 100
            else:
                performance_change = 0
            
            robustness_metrics = {
                'baseline_mAP_50': baseline_map50,
                'current_mAP_50': current_map50,
                'performance_change_percent': round(performance_change, 2),
                'robustness_category': self._categorize_robustness(performance_change),
                'degradation_analysis': {
                    'improved': performance_change > 0,
                    'degraded': performance_change < -5,  # >5% degradation considered significant
                    'stable': abs(performance_change) <= 5
                }
            }
            
            self.results['robustness'] = robustness_metrics
            
            self.logger.info(f"[ROBUSTNESS] Performance change: {performance_change:.2f}%")
            self.logger.info(f"[ROBUSTNESS] Category: {robustness_metrics['robustness_category']}")
            
            return robustness_metrics
            
        except Exception as e:
            self.logger.error(f"[ERROR] Robustness evaluation failed: {e}")
            return {}
    
    def _categorize_robustness(self, performance_change: float) -> str:
        """Categorize robustness based on performance change"""
        if performance_change > 10:
            return "significantly_improved"
        elif performance_change > 5:
            return "improved"
        elif performance_change > -5:
            return "stable"
        elif performance_change > -15:
            return "degraded"
        else:
            return "significantly_degraded"
    
    def generate_methodology_report(self, baseline_metrics: Optional[Dict] = None) -> str:
        """
        Generate comprehensive evaluation report according to thesis methodology
        """
        self.logger.info("[REPORT] Generating methodology compliance report...")
        
        report_file = self.output_dir / f"methodology_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# YOLOv8n Evaluation Metrics Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model**: {self.model_path}\n")
            f.write(f"**Dataset**: {self.dataset_config}\n\n")
            
            # Section 4.1 - Evaluation Metrics Compliance
            f.write("## Methodology Compliance (Section 4.1)\n\n")
            
            # Detection Accuracy
            acc_metrics = self.results.get('detection_accuracy', {})
            f.write("### Detection Accuracy\n")
            f.write(f"- **mAP@0.5**: {acc_metrics.get('mAP_50', 'N/A'):.4f}\n")
            f.write(f"- **mAP@0.5:0.95**: {acc_metrics.get('mAP_50_95', 'N/A'):.4f}\n")
            f.write(f"- **Precision**: {acc_metrics.get('precision', 'N/A'):.4f}\n")
            f.write(f"- **Recall**: {acc_metrics.get('recall', 'N/A'):.4f}\n")
            f.write(f"- **F1-Score**: {acc_metrics.get('f1_score', 'N/A'):.4f}\n\n")
            
            # Inference Speed
            speed_metrics = self.results.get('inference_speed', {})
            f.write("### Inference Speed\n")
            f.write(f"- **FPS**: {speed_metrics.get('fps', 'N/A')}\n")
            f.write(f"- **Average Inference Time**: {speed_metrics.get('avg_inference_time_ms', 'N/A')} ms\n")
            f.write(f"- **Min/Max Inference Time**: {speed_metrics.get('min_inference_time_ms', 'N/A')} / {speed_metrics.get('max_inference_time_ms', 'N/A')} ms\n\n")
            
            # Model Size
            size_metrics = self.results.get('model_size', {})
            f.write("### Model Size\n")
            f.write(f"- **File Size**: {size_metrics.get('model_file_size_mb', 'N/A')} MB\n")
            f.write(f"- **Memory Usage**: {size_metrics.get('memory_usage_mb', 'N/A')} MB\n")
            f.write(f"- **Parameters**: {size_metrics.get('total_parameters', 'N/A'):,}\n")
            f.write(f"- **Category**: {size_metrics.get('model_size_category', 'N/A')}\n\n")
            
            # Robustness (if baseline provided)
            if baseline_metrics:
                rob_metrics = self.results.get('robustness', {})
                f.write("### Robustness Analysis\n")
                f.write(f"- **Performance Change**: {rob_metrics.get('performance_change_percent', 'N/A'):.2f}%\n")
                f.write(f"- **Robustness Category**: {rob_metrics.get('robustness_category', 'N/A')}\n")
                f.write(f"- **Baseline mAP@0.5**: {rob_metrics.get('baseline_mAP_50', 'N/A'):.4f}\n")
                f.write(f"- **Current mAP@0.5**: {rob_metrics.get('current_mAP_50', 'N/A'):.4f}\n\n")
            
            # Hardware Information
            hardware = self.results.get('metadata', {}).get('hardware', {})
            f.write("### Hardware Configuration\n")
            f.write(f"- **GPU**: {hardware.get('gpu_name', 'N/A')} ({hardware.get('gpu_memory_gb', 'N/A')} GB)\n")
            f.write(f"- **CPU Cores**: {hardware.get('cpu_count', 'N/A')}\n")
            f.write(f"- **Memory**: {hardware.get('memory_total_gb', 'N/A')} GB\n")
            f.write(f"- **CUDA**: {hardware.get('cuda_version', 'N/A')}\n")
            f.write(f"- **PyTorch**: {hardware.get('pytorch_version', 'N/A')}\n\n")
        
        self.logger.info(f"[REPORT] Methodology report saved: {report_file}")
        return str(report_file)
    
    def save_results_json(self) -> str:
        """Save evaluation results to JSON for programmatic analysis"""
        json_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"[RESULTS] JSON results saved: {json_file}")
        return str(json_file)
    
    def run_comprehensive_evaluation(self, baseline_metrics: Optional[Dict] = None) -> Dict:
        """
        Run comprehensive evaluation according to thesis methodology
        """
        self.logger.info("[EVALUATION] Starting comprehensive evaluation...")
        
        # Run all evaluation components
        self.evaluate_detection_accuracy()
        self.evaluate_inference_speed()
        self.evaluate_model_size()
        
        if baseline_metrics:
            self.evaluate_robustness_degradation(baseline_metrics)
        
        # Generate reports
        self.generate_methodology_report(baseline_metrics)
        self.save_results_json()
        
        self.logger.info("[EVALUATION] Comprehensive evaluation completed!")
        return self.results


def load_baseline_metrics(baseline_results_file: str) -> Dict:
    """Load baseline metrics from JSON file for comparison"""
    try:
        with open(baseline_results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load baseline metrics: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8n Comprehensive Evaluation")
    parser.add_argument('--model', required=True, help='Path to trained model weights')
    parser.add_argument('--data', required=True, help='Path to dataset configuration')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--baseline', help='Path to baseline results JSON for comparison')
    
    args = parser.parse_args()
    
    # Load baseline metrics if provided
    baseline_metrics = None
    if args.baseline:
        baseline_metrics = load_baseline_metrics(args.baseline)
    
    # Run evaluation
    evaluator = YOLOv8nEvaluationMetrics(
        model_path=args.model,
        dataset_config=args.data,
        output_dir=Path(args.output)
    )
    
    results = evaluator.run_comprehensive_evaluation(baseline_metrics)
    print(f"\n[SUCCESS] Evaluation completed! Results saved to: {args.output}")