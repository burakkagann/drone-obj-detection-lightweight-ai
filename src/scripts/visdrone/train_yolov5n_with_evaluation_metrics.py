#!/usr/bin/env python3
"""
YOLOv5n VisDrone Training with Comprehensive Training Metrics
Integrates training performance evaluation framework with YOLOv5 training pipeline.
"""

import sys
import os
import warnings
from pathlib import Path
import torch

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir / ".." / ".." / ".."
sys.path.append(str(project_root))

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Import YOLOv5 components
yolov5_dir = project_root / "src" / "models" / "YOLOv5"
sys.path.append(str(yolov5_dir))

try:
    from models.yolo import Model
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset, colorstr
    from utils.torch_utils import select_device
    import val as validate
except ImportError as e:
    print(f"❌ Error importing YOLOv5 components: {e}")
    print(f"📁 Make sure YOLOv5 is properly installed in: {yolov5_dir}")
    sys.exit(1)

# Import training metrics framework
from src.evaluation.thesis_metrics import ThesisMetricsCollector, VISDRONE_CLASSES

class YOLOv5TrainingEvaluator:
    """YOLOv5 trainer with integrated training metrics collection"""
    
    def __init__(self, 
                 data_config: str,
                 model_config: str = "models/yolov5n.yaml",
                 weights: str = "yolov5n.pt",
                 device: str = "0",
                 batch_size: int = 16,
                 img_size: int = 640):
        
        self.data_config = data_config
        self.model_config = model_config
        self.weights = weights
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Device setup
        self.device = select_device(device, batch_size=batch_size)
        
        # Initialize training metrics collector
        self.metrics_collector = ThesisMetricsCollector(
            class_names=VISDRONE_CLASSES,
            results_dir="results/training_metrics",
            device=str(self.device)
        )
        
        print(f"🎯 YOLOv5 Training Evaluator Initialized")
        print(f"📊 Device: {self.device}")
        print(f"📁 Data Config: {data_config}")
        print(f"🏗️ Model: {model_config}")

    def setup_model_and_data(self):
        """Setup model and data loaders"""
        print("🔧 Setting up model and data loaders...")
        
        # Load model
        self.model = Model(self.model_config, ch=3, nc=len(VISDRONE_CLASSES))
        
        # Load pretrained weights if specified
        if self.weights and self.weights != '':
            print(f"📦 Loading pretrained weights: {self.weights}")
            ckpt = torch.load(self.weights, map_location=self.device, weights_only=False)
            state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt
            
            # Filter out detection head layers that have size mismatch
            # (COCO has 80 classes, VisDrone has 10 classes)
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"⚠️  Skipping layer {k} due to size mismatch: {v.shape} vs {model_state_dict[k].shape if k in model_state_dict else 'missing'}")
            
            # Load compatible weights
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded {len(filtered_state_dict)}/{len(state_dict)} pretrained weights")
        
        self.model.to(self.device)
        
        # Setup data
        data_dict = check_dataset(self.data_config)
        
        # Create validation dataloader
        self.val_loader, _ = create_dataloader(
            data_dict['val'],
            self.img_size,
            self.batch_size,
            32,  # grid size
            single_cls=False,
            hyp=None,
            augment=False,
            cache=True,
            pad=0.5,
            rect=True,
            workers=4,
            prefix=colorstr('val: ')
        )
        
        print(f"✅ Model and data setup complete")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")

    def run_validation_with_metrics(self, epoch: int = 0, model_path: str = None):
        """Run validation and collect comprehensive training metrics"""
        print(f"\n📊 Running validation with training metrics (Epoch {epoch})...")
        
        # Setup data dictionary for validation
        data_dict = check_dataset(self.data_config)
        
        # Run YOLOv5 validation
        yolov5_results = validate.run(
            data=data_dict,  # Pass dictionary instead of string path
            weights=model_path if model_path else self.weights,
            batch_size=self.batch_size,
            imgsz=self.img_size,
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            verbose=True,
            plots=False,
            save_json=False
        )
        
        # Extract validation metrics from YOLOv5 results
        # yolov5_results structure: (metrics_tuple, maps, timing)
        # metrics_tuple: (mp, mr, map50, map, loss1, loss2, loss3)
        metrics_tuple = yolov5_results[0]
        validation_metrics = {
            'metrics/mAP_0.5': metrics_tuple[2],        # map50
            'metrics/mAP_0.5:0.95': metrics_tuple[3],   # map
            'metrics/precision': metrics_tuple[0],      # mp (mean precision)
            'metrics/recall': metrics_tuple[1],         # mr (mean recall)
        }
        
        # Print validation results for visibility
        print(f"📊 Validation Results:")
        print(f"   mAP@0.5: {metrics_tuple[2]:.6f}")
        print(f"   mAP@0.5:0.95: {metrics_tuple[3]:.6f}")
        print(f"   Precision: {metrics_tuple[0]:.6f}")
        print(f"   Recall: {metrics_tuple[1]:.6f}")
        
        # Collect comprehensive training metrics
        training_metrics = self.metrics_collector.collect_comprehensive_metrics(
            model=self.model,
            dataloader=self.val_loader,
            yolov5_results=validation_metrics,
            epoch=epoch,
            model_path=model_path
        )
        
        return training_metrics

    def benchmark_inference_only(self, num_samples: int = 100):
        """Run inference-only benchmarking for training requirements"""
        print(f"\n⚡ Running inference-only benchmark...")
        
        # Collect inference metrics
        fps, inference_time, memory_usage = self.metrics_collector.benchmark_inference_speed(
            model=self.model,
            dataloader=self.val_loader,
            num_samples=num_samples,
            warmup_runs=10
        )
        
        # Get system resource metrics
        gpu_power, gpu_util, cpu_util = self.metrics_collector.monitor_system_resources()
        
        # Get model characteristics
        model_size, parameters, flops = self.metrics_collector.get_model_characteristics(
            model=self.model,
            model_path=self.weights
        )
        
        # Print training-ready summary
        print(f"\n🎯 **TRAINING INFERENCE BENCHMARK RESULTS**")
        print("=" * 55)
        print(f"⚡ **Real-Time Performance:**")
        print(f"   FPS (Frames/sec):    {fps:.2f}")
        print(f"   Inference Time:      {inference_time:.2f} ms")
        print(f"   Memory Usage:        {memory_usage:.2f} MB")
        
        print(f"\n🔋 **Resource Efficiency:**")
        print(f"   GPU Power:           {gpu_power:.1f} W")
        print(f"   GPU Utilization:     {gpu_util:.1f}%")
        print(f"   CPU Utilization:     {cpu_util:.1f}%")
        
        print(f"\n📦 **Model Characteristics:**")
        print(f"   Model Size:          {model_size:.2f} MB")
        print(f"   Parameters:          {parameters:,}")
        print(f"   FLOPs:              {flops:.2f}B")
        
        print(f"\n✅ **Edge Device Readiness Assessment:**")
        edge_ready = "✅ READY" if fps >= 15 and model_size <= 50 else "⚠️ NEEDS OPTIMIZATION"
        print(f"   Status: {edge_ready}")
        print(f"   Reasoning:")
        print(f"     - FPS: {fps:.1f} ({'✅' if fps >= 15 else '❌'} Target: ≥15)")
        print(f"     - Size: {model_size:.1f}MB ({'✅' if model_size <= 50 else '❌'} Target: ≤50MB)")
        print("=" * 55)
        
        return {
            'fps': fps,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_usage,
            'gpu_power_watts': gpu_power,
            'gpu_utilization': gpu_util,
            'cpu_utilization': cpu_util,
            'model_size_mb': model_size,
            'parameters': parameters,
            'flops': flops
        }

    def generate_training_documentation(self):
        """Generate comprehensive training documentation"""
        print("\n📄 Generating training documentation...")
        
        # Save metrics to CSV
        csv_path = self.metrics_collector.save_metrics_to_csv()
        
        # Create performance plots
        self.metrics_collector.create_performance_plots()
        
        # Generate training report
        report = self.metrics_collector.generate_training_report()
        
        print(f"📊 Training documentation generated:")
        print(f"   📄 Report: Available in results/training_metrics/")
        print(f"   📈 Plots: Performance visualization saved")
        print(f"   📊 Data: {csv_path}")

def main():
    """Main execution function"""
    print("🚀 YOLOv5n VisDrone Training with Training Metrics")
    print("=" * 60)
    
    # Configuration
    data_config = "../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml"
    model_config = "models/yolov5n.yaml"
    weights = "yolov5n.pt"
    
    # Initialize trainer
    trainer = YOLOv5TrainingEvaluator(
        data_config=data_config,
        model_config=model_config,
        weights=weights,
        device="0",  # CUDA GPU
        batch_size=16,
        img_size=640
    )
    
    # Setup model and data
    trainer.setup_model_and_data()
    
    # Run validation with comprehensive metrics
    print("\n📊 Phase 1: Validation with Training Metrics")
    training_metrics = trainer.run_validation_with_metrics(epoch=0)
    
    # Run inference-only benchmark
    print("\n⚡ Phase 2: Inference Benchmarking")
    inference_results = trainer.benchmark_inference_only(num_samples=100)
    
    # Generate training documentation
    print("\n📄 Phase 3: Documentation Generation")
    trainer.generate_training_documentation()
    
    print("\n✅ Training metrics collection completed!")
    print("📊 Check results/training_metrics/ for comprehensive evaluation data")

if __name__ == "__main__":
    # Change to YOLOv5 directory for proper imports
    yolov5_dir = Path(__file__).parent / ".." / ".." / "models" / "YOLOv5"
    os.chdir(yolov5_dir)
    
    main() 