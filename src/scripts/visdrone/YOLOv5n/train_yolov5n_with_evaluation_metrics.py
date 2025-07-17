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
import yaml
from copy import deepcopy
import argparse

# Add project root to path
script_dir = Path(__file__).parent
print(f"[DEBUG] Script directory: {script_dir}")
project_root = script_dir / ".." / ".." / ".." / ".."  # Go up 4 levels to reach project root
project_root = project_root.resolve()  # Convert to absolute path
print(f"[DEBUG] Project root: {project_root}")
sys.path.append(str(project_root))

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Import YOLOv5 components
# Direct path from script location to YOLOv5
yolov5_dir = script_dir / ".." / ".." / ".." / "models" / "YOLOv5"
yolov5_dir = yolov5_dir.resolve()  # Convert to absolute path
print(f"[PATH] YOLOv5 directory: {yolov5_dir}")
print(f"[PATH] Directory exists: {yolov5_dir.exists()}")

# Change to YOLOv5 directory before importing
original_cwd = os.getcwd()
if yolov5_dir.exists():
    os.chdir(yolov5_dir)
    print(f"[PATH] Changed to YOLOv5 directory: {os.getcwd()}")
else:
    print(f"[ERROR] YOLOv5 directory does not exist: {yolov5_dir}")
    sys.exit(1)

sys.path.append(str(yolov5_dir))

try:
    from models.yolo import Model
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset, colorstr, increment_path, one_cycle
    from utils.torch_utils import select_device, ModelEMA
    from utils.loss import ComputeLoss
    from utils.plots import plot_labels, plot_evolve, plot_results
    from utils.loggers import Loggers
    from utils.callbacks import Callbacks
    import val as validate
    import train as yolov5_train
    print("[SUCCESS] YOLOv5 components imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing YOLOv5 components: {e}")
    print(f"[PATH] Make sure YOLOv5 is properly installed in: {yolov5_dir}")
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
                 img_size: int = 640,
                 epochs: int = 100,
                 save_period: int = 5,
                 patience: int = 10,
                 project: str = "runs/train",
                 name: str = "yolov5n_visdrone"):
        
        self.data_config = data_config
        self.model_config = model_config
        self.weights = weights
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.save_period = save_period
        self.patience = patience
        self.project = project
        self.name = name
        
        # Device setup
        self.device = select_device(device, batch_size=batch_size)
        
        # Initialize training metrics collector
        self.metrics_collector = ThesisMetricsCollector(
            class_names=VISDRONE_CLASSES,
            results_dir="results/training_metrics",
            device=str(self.device)
        )
        
        # Create save directory
        self.save_dir = increment_path(Path(project) / name, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INIT] YOLOv5 Training Evaluator Initialized")
        print(f"[DEVICE] Device: {self.device}")
        print(f"[DATA] Data Config: {data_config}")
        print(f"[MODEL] Model: {model_config}")
        print(f"[SAVE] Save Directory: {self.save_dir}")

    def setup_model_and_data(self):
        """Setup model and data loaders"""
        print("[SETUP] Setting up model and data loaders...")
        
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
                    print(f"[WARNING] Skipping layer {k} due to size mismatch: {v.shape} vs {model_state_dict[k].shape if k in model_state_dict else 'missing'}")
            
            # Load compatible weights
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"[LOADED] Loaded {len(filtered_state_dict)}/{len(state_dict)} pretrained weights")
        
        self.model.to(self.device)
        
        # Setup data
        self.data_dict = check_dataset(self.data_config)
        
        # Create data loaders
        self.train_loader, self.train_dataset = create_dataloader(
            self.data_dict['train'],
            self.img_size,
            self.batch_size,
            32,  # grid size
            single_cls=False,
            hyp=self.get_hyperparameters(),
            augment=True,
            cache=False,
            pad=0.0,
            rect=False,
            workers=4,
            prefix=colorstr('train: ')
        )
        
        self.val_loader, self.val_dataset = create_dataloader(
            self.data_dict['val'],
            self.img_size,
            self.batch_size,
            32,  # grid size
            single_cls=False,
            hyp=self.get_hyperparameters(),
            augment=False,
            cache=False,
            pad=0.5,
            rect=True,
            workers=4,
            prefix=colorstr('val: ')
        )
        
        print(f"[SUCCESS] Model and data setup complete")
        print(f"[DATA] Training samples: {len(self.train_dataset)}")
        print(f"[DATA] Validation samples: {len(self.val_dataset)}")

    def get_hyperparameters(self):
        """Get default hyperparameters for training"""
        return {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }

    def train_with_metrics(self):
        """Train YOLOv5 with integrated metrics collection"""
        print(f"\n[START] Starting training with {self.epochs} epochs...")
        
        # Setup training components
        hyp = self.get_hyperparameters()
        
        # Attach hyperparameters to model (required for ComputeLoss)
        self.model.hyp = hyp
        
        # Optimizer
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k)
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, bn):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                g[0].append(v.weight)
        
        optimizer = torch.optim.SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        
        # Scheduler
        lf = one_cycle(1, hyp['lrf'], self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        # Loss function
        compute_loss = ComputeLoss(self.model)
        
        # EMA
        ema = ModelEMA(self.model)
        
        # Training loop
        best_fitness = 0.0
        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"🏃 Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*60}")
            
            # Training phase
            train_loss = self.train_epoch(epoch, optimizer, scheduler, compute_loss, ema)
            
            # Validation phase with metrics
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                val_metrics = self.validate_with_metrics(epoch, ema.ema)
                
                # Save best model
                fitness = val_metrics.get('metrics/mAP_0.5', 0.0)
                if fitness > best_fitness:
                    best_fitness = fitness
                    torch.save({
                        'epoch': epoch,
                        'model': deepcopy(ema.ema).half(),
                        'optimizer': optimizer.state_dict(),
                        'best_fitness': best_fitness
                    }, self.save_dir / 'best.pt')
                    print(f"🏆 New best model saved with mAP@0.5: {fitness:.4f}")
                
                # Save checkpoint
                if epoch % self.save_period == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': deepcopy(ema.ema).half(),
                        'optimizer': optimizer.state_dict(),
                        'best_fitness': best_fitness
                    }, self.save_dir / f'epoch_{epoch}.pt')
        
        print(f"\n[SUCCESS] Training completed! Best mAP@0.5: {best_fitness:.4f}")
        return self.save_dir / 'best.pt'

    def train_epoch(self, epoch, optimizer, scheduler, compute_loss, ema):
        """Train one epoch"""
        self.model.train()
        
        pbar = enumerate(self.train_loader)
        print(f"[EPOCH] Training epoch {epoch+1}")
        
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device)
            
            # Forward pass
            pred = self.model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update(self.model)
            
            if i % 50 == 0:
                print(f"   Batch {i}/{len(self.train_loader)}: Loss {loss.item():.4f}")
        
        scheduler.step()
        return loss.item()

    def validate_with_metrics(self, epoch, model=None):
        """Validate and collect comprehensive metrics"""
        print(f"\n[VALIDATION] Running validation with training metrics (Epoch {epoch+1})...")
        
        # Use EMA model if provided
        val_model = model if model is not None else self.model
        
        # Run YOLOv5 validation
        yolov5_results = validate.run(
            data=self.data_dict,
            weights=None,
            batch_size=self.batch_size,
            imgsz=self.img_size,
            model=val_model,
            dataloader=self.val_loader,
            device=self.device,
            verbose=False,
            plots=False,
            save_json=False
        )
        
        # Extract validation metrics
        metrics_tuple = yolov5_results[0]
        validation_metrics = {
            'metrics/mAP_0.5': metrics_tuple[2],
            'metrics/mAP_0.5:0.95': metrics_tuple[3],
            'metrics/precision': metrics_tuple[0],
            'metrics/recall': metrics_tuple[1],
        }
        
        # Print validation results
        print(f"[RESULTS] Validation Results (Epoch {epoch+1}):")
        print(f"   mAP@0.5: {metrics_tuple[2]:.6f}")
        print(f"   mAP@0.5:0.95: {metrics_tuple[3]:.6f}")
        print(f"   Precision: {metrics_tuple[0]:.6f}")
        print(f"   Recall: {metrics_tuple[1]:.6f}")
        
        # Collect comprehensive training metrics
        training_metrics = self.metrics_collector.collect_comprehensive_metrics(
            model=val_model,
            dataloader=self.val_loader,
            yolov5_results=validation_metrics,
            epoch=epoch+1,
            model_path=str(self.save_dir / f'epoch_{epoch}.pt')
        )
        
        return validation_metrics

    def run_validation_with_metrics(self, epoch: int = 0, model_path: str = None):
        """Run validation and collect comprehensive training metrics"""
        print(f"\n[VALIDATION] Running validation with training metrics (Epoch {epoch})...")
        
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
        print(f"[RESULTS] Validation Results:")
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
        print(f"\n[BENCHMARK] Running inference-only benchmark...")
        
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
        print(f"\n[RESULTS] **TRAINING INFERENCE BENCHMARK RESULTS**")
        print("=" * 55)
        print(f"[PERF] **Real-Time Performance:**")
        print(f"   FPS (Frames/sec):    {fps:.2f}")
        print(f"   Inference Time:      {inference_time:.2f} ms")
        print(f"   Memory Usage:        {memory_usage:.2f} MB")
        
        print(f"\n[RESOURCE] **Resource Efficiency:**")
        print(f"   GPU Power:           {gpu_power:.1f} W")
        print(f"   GPU Utilization:     {gpu_util:.1f}%")
        print(f"   CPU Utilization:     {cpu_util:.1f}%")
        
        print(f"\n[MODEL] **Model Characteristics:**")
        print(f"   Model Size:          {model_size:.2f} MB")
        print(f"   Parameters:          {parameters:,}")
        print(f"   FLOPs:              {flops:.2f}B")
        
        print(f"\n[EDGE] **Edge Device Readiness Assessment:**")
        edge_ready = "[READY]" if fps >= 15 and model_size <= 50 else "[NEEDS OPTIMIZATION]"
        print(f"   Status: {edge_ready}")
        print(f"   Reasoning:")
        print(f"     - FPS: {fps:.1f} ({'[PASS]' if fps >= 15 else '[FAIL]'} Target: ≥15)")
        print(f"     - Size: {model_size:.1f}MB ({'[PASS]' if model_size <= 50 else '[FAIL]'} Target: ≤50MB)")
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
        
        print(f"[DOCS] Training documentation generated:")
        print(f"   [REPORT] Report: Available in results/training_metrics/")
        print(f"   [PLOTS] Plots: Performance visualization saved")
        print(f"   [DATA] Data: {csv_path}")

def main():
    """Main execution function"""
    print("[START] YOLOv5n VisDrone Training with Training Metrics")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Training with Metrics')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=416, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--name', type=str, default='yolov5n_visdrone', help='Experiment name')
    parser.add_argument('--data', type=str, default='../../../data/my_dataset/visdrone_augmented/dataset_config.yaml', help='Dataset configuration file')
    args = parser.parse_args()
    
    # Configuration - Use augmented dataset if available, fallback to original
    # Paths are now relative to YOLOv5 directory
    augmented_data_config = "../../../data/my_dataset/visdrone_augmented/dataset_config.yaml"
    original_data_config = "../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml"
    
    if Path(args.data).exists():
        data_config = args.data
        print(f"[DATA] Using augmented dataset: {data_config}")
    elif Path(augmented_data_config).exists():
        data_config = augmented_data_config
        print(f"[DATA] Using augmented dataset: {data_config}")
    else:
        data_config = original_data_config
        print(f"[DATA] Using original dataset: {data_config}")
        print("[WARNING] Augmented dataset not found, using original dataset")
    model_config = "models/yolov5n.yaml"
    weights = "yolov5n.pt"
    
    # Initialize trainer with parsed arguments
    trainer = YOLOv5TrainingEvaluator(
        data_config=data_config,
        model_config=model_config,
        weights=weights,
        device=args.device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        epochs=args.epochs,
        project="runs/train",
        name=args.name
    )
    
    # Setup model and data
    trainer.setup_model_and_data()
    
    # Option 1: Full training with metrics
    print(f"\n[START] Starting full training with integrated metrics for {args.epochs} epochs...")
    best_model_path = trainer.train_with_metrics()
    
    # Option 2: If you just want evaluation (comment out above and uncomment below)
    # print("\n[PHASE 1] Phase 1: Validation with Training Metrics")
    # training_metrics = trainer.run_validation_with_metrics(epoch=0)
    # 
    # print("\n[PHASE 2] Phase 2: Inference Benchmarking")
    # inference_results = trainer.benchmark_inference_only(num_samples=100)
    
    # Generate training documentation
    print("\n[DOCS] Final Documentation Generation")
    trainer.generate_training_documentation()
    
    print("\n[SUCCESS] Training with metrics collection completed!")
    print("[RESULTS] Check results/training_metrics/ for comprehensive evaluation data")
    print(f"[BEST] Best model saved at: {best_model_path}")

if __name__ == "__main__":
    # Directory change is now handled at the top before imports
    main() 