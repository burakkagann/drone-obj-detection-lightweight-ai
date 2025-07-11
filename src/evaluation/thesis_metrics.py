#!/usr/bin/env python3
"""
Thesis Performance Metrics Evaluation Framework
Comprehensive metrics collection for YOLOv5n VisDrone training aligned with thesis methodology.

Metrics Implemented:
- Detection Accuracy: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Real-Time Performance: FPS, inference time, memory usage
- Power Consumption: GPU power monitoring
- Model Characteristics: size, parameters, FLOPS
"""

import time
import json
import psutil
import GPUtil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Data class to store all performance metrics"""
    # Detection Accuracy Metrics
    map_50: float = 0.0           # mAP@0.5
    map_50_95: float = 0.0        # mAP@0.5:0.95
    precision: float = 0.0        # Overall precision
    recall: float = 0.0           # Overall recall
    
    # Per-class metrics
    class_map_50: Dict[str, float] = None
    class_precision: Dict[str, float] = None
    class_recall: Dict[str, float] = None
    
    # Real-Time Performance
    fps: float = 0.0              # Frames per second
    inference_time_ms: float = 0.0 # Average inference time
    memory_usage_mb: float = 0.0   # GPU memory usage
    
    # Power and System
    gpu_power_watts: float = 0.0   # GPU power consumption
    gpu_utilization: float = 0.0   # GPU utilization %
    cpu_utilization: float = 0.0   # CPU utilization %
    
    # Model Characteristics
    model_size_mb: float = 0.0     # Model file size
    parameters: int = 0            # Number of parameters
    flops: float = 0.0            # FLOPs (billions)
    
    # Training Context
    epoch: int = 0
    batch_size: int = 16
    image_size: int = 640
    dataset: str = "visdrone"
    device: str = "cuda"
    timestamp: str = ""

class ThesisMetricsCollector:
    """Comprehensive metrics collector for thesis evaluation"""
    
    def __init__(self, 
                 class_names: List[str],
                 results_dir: str = "results/training_metrics",
                 device: str = "cuda"):
        
        self.class_names = class_names
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Initialize metrics storage
        self.metrics_history = []
        self.class_metrics_history = defaultdict(list)
        
        # GPU monitoring setup
        self.gpus = GPUtil.getGPUs()
        self.gpu = self.gpus[0] if self.gpus else None
        
        print(f"📊 Training Metrics Collector Initialized")
        print(f"📁 Results Directory: {self.results_dir}")
        print(f"🎯 Classes: {len(class_names)} ({', '.join(class_names[:3])}...)")
        print(f"🖥️ GPU: {self.gpu.name if self.gpu else 'None'}")

    def collect_yolov5_validation_metrics(self, results: Dict) -> PerformanceMetrics:
        """
        Extract metrics from YOLOv5 validation results
        
        Args:
            results: YOLOv5 validation results dictionary
            
        Returns:
            PerformanceMetrics object with populated accuracy metrics
        """
        metrics = PerformanceMetrics()
        
        # Extract mAP metrics from YOLOv5 results
        if 'metrics/mAP_0.5' in results:
            metrics.map_50 = float(results['metrics/mAP_0.5'])
        if 'metrics/mAP_0.5:0.95' in results:
            metrics.map_50_95 = float(results['metrics/mAP_0.5:0.95'])
        if 'metrics/precision' in results:
            metrics.precision = float(results['metrics/precision'])
        if 'metrics/recall' in results:
            metrics.recall = float(results['metrics/recall'])
            
        # Per-class metrics (if available)
        if 'per_class_metrics' in results:
            class_metrics = results['per_class_metrics']
            metrics.class_map_50 = {name: float(class_metrics.get(f'{name}_mAP_0.5', 0.0)) 
                                   for name in self.class_names}
            metrics.class_precision = {name: float(class_metrics.get(f'{name}_precision', 0.0)) 
                                     for name in self.class_names}
            metrics.class_recall = {name: float(class_metrics.get(f'{name}_recall', 0.0)) 
                                  for name in self.class_names}
        
        return metrics

    def benchmark_inference_speed(self, 
                                 model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 num_samples: int = 100,
                                 warmup_runs: int = 10) -> Tuple[float, float, float]:
        """
        Benchmark inference speed and measure FPS
        
        Args:
            model: YOLOv5 model
            dataloader: Validation dataloader
            num_samples: Number of samples to benchmark
            warmup_runs: Number of warmup runs
            
        Returns:
            Tuple of (fps, avg_inference_time_ms, memory_usage_mb)
        """
        model.eval()
        
        # Warmup runs
        print(f"🔥 Warming up with {warmup_runs} runs...")
        with torch.no_grad():
            for i, (images, targets, paths, shapes) in enumerate(dataloader):
                if i >= warmup_runs:
                    break
                images = images.to(self.device).float() / 255.0  # Convert to float and normalize
                _ = model(images)
        
        # Clear cache and measure baseline memory
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        # Benchmark inference
        inference_times = []
        print(f"⏱️ Benchmarking inference speed with {num_samples} samples...")
        
        with torch.no_grad():
            for i, (images, targets, paths, shapes) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                images = images.to(self.device).float() / 255.0  # Convert to float and normalize
                
                # Measure inference time
                start_time = time.perf_counter()
                _ = model(images)
                
                # Synchronize GPU (important for accurate timing)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        fps = 1000.0 / avg_inference_time  # Convert ms to FPS
        
        # Memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        memory_usage = peak_memory - baseline_memory
        
        print(f"⚡ Inference Results:")
        print(f"   FPS: {fps:.2f}")
        print(f"   Avg Inference Time: {avg_inference_time:.2f} ms")
        print(f"   Memory Usage: {memory_usage:.2f} MB")
        
        return fps, avg_inference_time, memory_usage

    def monitor_system_resources(self) -> Tuple[float, float, float]:
        """
        Monitor system resources (GPU power, GPU utilization, CPU utilization)
        
        Returns:
            Tuple of (gpu_power_watts, gpu_utilization, cpu_utilization)
        """
        # GPU monitoring
        gpu_power = 0.0
        gpu_utilization = 0.0
        
        if self.gpu:
            # Refresh GPU stats
            GPUtil.showUtilization()
            self.gpu = GPUtil.getGPUs()[0]
            
            gpu_utilization = self.gpu.load * 100  # Convert to percentage
            
            # GPU power (if supported by nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except:
                gpu_power = 0.0  # Not available
        
        # CPU utilization
        cpu_utilization = psutil.cpu_percent(interval=1)
        
        return gpu_power, gpu_utilization, cpu_utilization

    def get_model_characteristics(self, model: torch.nn.Module, model_path: str = None) -> Tuple[float, int, float]:
        """
        Get model characteristics (size, parameters, FLOPs)
        
        Args:
            model: PyTorch model
            model_path: Path to model file for size calculation
            
        Returns:
            Tuple of (model_size_mb, parameters, flops_billions)
        """
        # Model size from file
        model_size_mb = 0.0
        if model_path and Path(model_path).exists():
            model_size_mb = Path(model_path).stat().st_size / 1024**2
        
        # Parameter count
        parameters = sum(p.numel() for p in model.parameters())
        
        # FLOPs calculation (simplified estimate)
        # For YOLOv5n, approximate FLOPs based on input size
        input_size = 640  # Assuming 640x640 input
        flops_billions = (parameters * input_size * input_size) / 1e9  # Rough estimate
        
        return model_size_mb, parameters, flops_billions

    def collect_comprehensive_metrics(self,
                                    model: torch.nn.Module,
                                    dataloader: torch.utils.data.DataLoader,
                                    yolov5_results: Dict = None,
                                    epoch: int = 0,
                                    model_path: str = None) -> PerformanceMetrics:
        """
        Collect all metrics in one comprehensive evaluation
        
        Args:
            model: YOLOv5 model
            dataloader: Validation dataloader
            yolov5_results: YOLOv5 validation results
            epoch: Current training epoch
            model_path: Path to model file
            
        Returns:
            Complete PerformanceMetrics object
        """
        print(f"\n📊 Collecting Comprehensive Metrics (Epoch {epoch})...")
        
        # Initialize metrics
        metrics = PerformanceMetrics()
        metrics.epoch = epoch
        metrics.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Detection Accuracy Metrics
        if yolov5_results:
            accuracy_metrics = self.collect_yolov5_validation_metrics(yolov5_results)
            metrics.map_50 = accuracy_metrics.map_50
            metrics.map_50_95 = accuracy_metrics.map_50_95
            metrics.precision = accuracy_metrics.precision
            metrics.recall = accuracy_metrics.recall
            metrics.class_map_50 = accuracy_metrics.class_map_50
            metrics.class_precision = accuracy_metrics.class_precision
            metrics.class_recall = accuracy_metrics.class_recall
        
        # 2. Inference Speed Benchmarking
        fps, inference_time, memory_usage = self.benchmark_inference_speed(model, dataloader)
        metrics.fps = fps
        metrics.inference_time_ms = inference_time
        metrics.memory_usage_mb = memory_usage
        
        # 3. System Resource Monitoring
        gpu_power, gpu_util, cpu_util = self.monitor_system_resources()
        metrics.gpu_power_watts = gpu_power
        metrics.gpu_utilization = gpu_util
        metrics.cpu_utilization = cpu_util
        
        # 4. Model Characteristics
        model_size, parameters, flops = self.get_model_characteristics(model, model_path)
        metrics.model_size_mb = model_size
        metrics.parameters = parameters
        metrics.flops = flops
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log summary
        self.log_metrics_summary(metrics)
        
        return metrics

    def log_metrics_summary(self, metrics: PerformanceMetrics):
        """Print a summary of collected metrics"""
        print(f"\n🎯 **TRAINING METRICS SUMMARY** (Epoch {metrics.epoch})")
        print("=" * 60)
        print(f"📊 **Detection Accuracy:**")
        print(f"   mAP@0.5:     {metrics.map_50:.4f}")
        print(f"   mAP@0.5:0.95: {metrics.map_50_95:.4f}")
        print(f"   Precision:   {metrics.precision:.4f}")
        print(f"   Recall:      {metrics.recall:.4f}")
        
        print(f"\n⚡ **Real-Time Performance:**")
        print(f"   FPS:         {metrics.fps:.2f}")
        print(f"   Inference:   {metrics.inference_time_ms:.2f} ms")
        print(f"   Memory:      {metrics.memory_usage_mb:.2f} MB")
        
        print(f"\n🔋 **System Resources:**")
        print(f"   GPU Power:   {metrics.gpu_power_watts:.1f} W")
        print(f"   GPU Usage:   {metrics.gpu_utilization:.1f}%")
        print(f"   CPU Usage:   {metrics.cpu_utilization:.1f}%")
        
        print(f"\n📦 **Model Characteristics:**")
        print(f"   Size:        {metrics.model_size_mb:.2f} MB")
        print(f"   Parameters:  {metrics.parameters:,}")
        print(f"   FLOPs:       {metrics.flops:.2f}B")
        print("=" * 60)

    def save_metrics_to_csv(self) -> str:
        """Save all collected metrics to CSV file"""
        if not self.metrics_history:
            print("⚠️ No metrics to save")
            return ""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"yolov5n_visdrone_metrics_{timestamp}.csv"
        
        # Convert metrics to DataFrame
        data = []
        for metrics in self.metrics_history:
            row = {
                'epoch': metrics.epoch,
                'timestamp': metrics.timestamp,
                'map_50': metrics.map_50,
                'map_50_95': metrics.map_50_95,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'fps': metrics.fps,
                'inference_time_ms': metrics.inference_time_ms,
                'memory_usage_mb': metrics.memory_usage_mb,
                'gpu_power_watts': metrics.gpu_power_watts,
                'gpu_utilization': metrics.gpu_utilization,
                'cpu_utilization': metrics.cpu_utilization,
                'model_size_mb': metrics.model_size_mb,
                'parameters': metrics.parameters,
                'flops': metrics.flops
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        print(f"💾 Metrics saved to: {csv_path}")
        return str(csv_path)

    def create_performance_plots(self):
        """Create performance visualization plots"""
        if not self.metrics_history:
            print("⚠️ No metrics to plot")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"performance_plots_{timestamp}.png"
        
        # Extract data for plotting
        epochs = [m.epoch for m in self.metrics_history]
        map_50 = [m.map_50 for m in self.metrics_history]
        fps = [m.fps for m in self.metrics_history]
        memory = [m.memory_usage_mb for m in self.metrics_history]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # mAP@0.5 over epochs
        ax1.plot(epochs, map_50, 'b-o')
        ax1.set_title('mAP@0.5 vs Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP@0.5')
        ax1.grid(True)
        
        # FPS over epochs
        ax2.plot(epochs, fps, 'g-o')
        ax2.set_title('FPS vs Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FPS')
        ax2.grid(True)
        
        # Memory usage over epochs
        ax3.plot(epochs, memory, 'r-o')
        ax3.set_title('Memory Usage vs Epochs')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Memory (MB)')
        ax3.grid(True)
        
        # Model characteristics
        latest = self.metrics_history[-1]
        ax4.bar(['mAP@0.5', 'FPS/10', 'Memory/100'], 
                [latest.map_50, latest.fps/10, latest.memory_usage_mb/100])
        ax4.set_title('Latest Performance Metrics')
        ax4.set_ylabel('Normalized Values')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to: {plot_path}")

    def generate_training_report(self) -> str:
        """Generate comprehensive training evaluation report"""
        if not self.metrics_history:
            return "No metrics collected yet."
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        latest_metrics = self.metrics_history[-1]
        
        report = f"""
# YOLOv5n VisDrone Training Evaluation Report

**Generated:** {latest_metrics.timestamp}
**Model:** YOLOv5n
**Dataset:** VisDrone (10 classes)
**Device:** {self.device}

## 📊 Detection Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | {latest_metrics.map_50:.4f} |
| mAP@0.5:0.95 | {latest_metrics.map_50_95:.4f} |
| Precision | {latest_metrics.precision:.4f} |
| Recall | {latest_metrics.recall:.4f} |

## ⚡ Real-Time Performance

| Metric | Value |
|--------|-------|
| FPS | {latest_metrics.fps:.2f} |
| Inference Time | {latest_metrics.inference_time_ms:.2f} ms |
| Memory Usage | {latest_metrics.memory_usage_mb:.2f} MB |

## 🔋 System Resources

| Metric | Value |
|--------|-------|
| GPU Power | {latest_metrics.gpu_power_watts:.1f} W |
| GPU Utilization | {latest_metrics.gpu_utilization:.1f}% |
| CPU Utilization | {latest_metrics.cpu_utilization:.1f}% |

## 📦 Model Characteristics

| Metric | Value |
|--------|-------|
| Model Size | {latest_metrics.model_size_mb:.2f} MB |
| Parameters | {latest_metrics.parameters:,} |
| FLOPs | {latest_metrics.flops:.2f}B |

## ✅ Edge Device Assessment

**Target Requirements:**
- FPS: ≥15 for real-time detection
- Model Size: ≤50MB for edge deployment

**Current Status:**
- FPS: {latest_metrics.fps:.1f} ({'✅ PASS' if latest_metrics.fps >= 15 else '❌ NEEDS OPTIMIZATION'})
- Size: {latest_metrics.model_size_mb:.1f}MB ({'✅ PASS' if latest_metrics.model_size_mb <= 50 else '❌ NEEDS OPTIMIZATION'})

## 📝 Training Recommendations

{self._generate_recommendations(latest_metrics)}

---
*This report was automatically generated by the YOLOv5 Training Evaluation Framework*
"""
        
        # Save report to file
        report_path = Path(self.results_dir) / f"training_report_{timestamp}.md"
        report_path.write_text(report, encoding='utf-8')  # Specify UTF-8 encoding
        
        print(f"📄 Training report saved to: {report_path}")
        return str(report_path)

    def _generate_recommendations(self, metrics: PerformanceMetrics) -> str:
        """Generate training recommendations based on metrics"""
        recommendations = []
        
        if metrics.fps < 15:
            recommendations.append("- **Optimize for Speed:** Consider model pruning, quantization, or TensorRT optimization")
        
        if metrics.map_50 < 0.5:
            recommendations.append("- **Improve Accuracy:** Continue training, adjust learning rate, or use data augmentation")
        
        if metrics.memory_usage_mb > 500:
            recommendations.append("- **Reduce Memory:** Consider mixed precision training or smaller batch sizes")
        
        if metrics.model_size_mb > 50:
            recommendations.append("- **Model Compression:** Apply model compression techniques for edge deployment")
        
        if not recommendations:
            recommendations.append("- **Performance Goals Met:** Model is ready for deployment")
        
        return "\n".join(recommendations)

# VisDrone class names for reference
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

if __name__ == "__main__":
    # Example usage
    collector = ThesisMetricsCollector(
        class_names=VISDRONE_CLASSES,
        results_dir="results/training_metrics"
    )
    
    print("🎯 Training Metrics Collector Ready!")
    print("📝 Use this collector during YOLOv5 training to gather comprehensive metrics.")
    print("📊 Call collect_comprehensive_metrics() after each epoch or validation run.") 