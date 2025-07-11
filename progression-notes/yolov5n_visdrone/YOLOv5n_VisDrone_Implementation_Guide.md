# YOLOv5n Implementation Guide for VisDrone Dataset

## Step 1: Environment Setup and Memory Optimization

### 1.1 Initial Setup
```bash
# Create and activate virtual environment
python -m venv venvs/yolov5n_visdrone_env
source venvs/yolov5n_visdrone_env/bin/activate  # Linux/Mac
# or
.\venvs\yolov5n_visdrone_env\Scripts\activate  # Windows

# Install dependencies with specific versions
pip install torch==2.0.1 torchvision==0.15.2
pip install -r src/models/YOLOv5/requirements.txt
```

### 1.2 Memory Management Setup
```python
# Add to train.py
import gc
import psutil
import torch

def setup_memory_management():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()
    
    # Disable gradient calculation for validation
    torch.set_grad_enabled(False)
    
    # Monitor memory
    print(f"CPU Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
```

## Step 2: Dataset Preparation

### 2.1 VisDrone Data Structure
```
data/
└── visdrone/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### 2.2 Modified Dataset Loading
```python
# Create custom_dataset.py
from torch.utils.data import Dataset
import numpy as np

class VisDroneDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = self._load_image_list()
        
    def _load_image_list(self):
        # Implement progressive loading
        return [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Implement memory-efficient loading
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, 
                                 self.image_files[idx].replace('.jpg', '.txt'))
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        labels = np.loadtxt(label_path).reshape(-1, 5)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels
```

## Step 3: Model Configuration

### 3.1 YOLOv5n Configuration
Create `config/visdrone/yolov5n_visdrone.yaml`:
```yaml
# YOLOv5n configuration
nc: 10  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.25  # layer channel multiple

# Anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5n backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5n head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

### 3.2 Training Configuration
```python
# Modified training configuration
hyp = {
    'lr0': 0.01,
    'lrf': 0.1,
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
    'mosaic': 0.0,  # Disable mosaic augmentation
    'mixup': 0.0,   # Disable mixup augmentation
    'copy_paste': 0.0  # Disable copy-paste augmentation
}
```

## Step 4: Memory-Efficient Training Implementation

### 4.1 Modified Anchor Computation
```python
# Modify utils/autoanchor.py
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    prefix = colorstr('AutoAnchor: ')
    print(f'{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1. / thr).float().sum(1).mean()
        bpr = (best > 1. / thr).float().mean()
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}')
    if bpr < 0.98:
        print(f'{prefix}Attempting to improve anchors...')
        na = m.anchor_grid.numel() // 2
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)
            check_anchor_order(m)
            print(f'{prefix}New anchors saved to model. Update model config to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
```

### 4.2 Training Loop Modifications
```python
# Modify train.py
def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights

    # Setup
    setup_memory_management()
    
    # Dataset
    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Dataloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz=opt.imgsz,
        batch_size=batch_size // opt.world_size,
        stride=int(model.stride.max()),
        hyp=hyp,
        augment=True,
        cache=None,  # Disable caching
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=opt.workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True
    )
    
    # Training
    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Memory cleanup before epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        pbar = enumerate(train_loader)
        for i, (imgs, targets, paths, _) in pbar:
            # Batch accumulation
            if i % accumulate == 0:
                optimizer.zero_grad()
            
            # Forward
            loss = model(imgs, targets)
            
            # Backward
            loss.backward()
            
            # Optimize
            if i % accumulate == accumulate - 1:
                optimizer.step()
            
            # Memory cleanup in batch loop
            del imgs, targets
            if i % 10 == 0:  # Every 10 batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
```

## Step 5: Evaluation and Monitoring

### 5.1 Memory-Aware Validation
```python
# Add to validation.py
def memory_aware_validation(model, dataloader, compute_loss):
    training = model.training
    model.eval()
    
    # Initialize metrics
    stats = []
    
    with torch.no_grad():
        for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
            # Inference
            out, train_out = model(im) if compute_loss else (model(im), None)
            
            # Compute loss
            if compute_loss:
                loss = compute_loss([x.float() for x in train_out], targets)[1]
            
            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True)
            
            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                stats.append((pred, labels, tcls))
            
            # Memory cleanup
            del im, targets, out
            if batch_i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    model.train(training)
    return stats
```

### 5.2 Performance Monitoring
```python
# Add to utils/metrics.py
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metric_name, value):
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name):
        return np.mean(self.metrics[metric_name])
    
    def log_memory_usage(self):
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.update('cpu_memory', cpu_memory)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            self.update('gpu_memory', gpu_memory)
    
    def log_batch_metrics(self, batch_time, loss):
        self.update('batch_time', batch_time)
        self.update('loss', loss)
```

## Step 6: Running the Training

### 6.1 Training Command
```bash
python train.py \
    --img 416 \
    --batch 8 \
    --epochs 100 \
    --data config/visdrone/visdrone.yaml \
    --cfg config/visdrone/yolov5n_visdrone.yaml \
    --weights yolov5n.pt \
    --device 0 \
    --workers 2 \
    --cache False
```

### 6.2 Monitoring Command
```bash
# In a separate terminal
watch -n 1 nvidia-smi  # For GPU monitoring
top  # For CPU monitoring
```

Would you like me to proceed with implementing any specific part of this guide? 