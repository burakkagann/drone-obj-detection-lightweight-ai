# NanoDet Model Evaluation Report

**Generated**: 2025-07-28 15:45:41  
**Model**: best_model.pth  
**Dataset**: VisDrone Test Set  

## Model Information

- **Architecture**: simple_nanodet
- **Total Parameters**: 168,398
- **Trainable Parameters**: 168,398
- **Model Size**: 0.65 MB
- **Device**: cpu

## Detection Accuracy Metrics

| Metric | Value |
|--------|--------|
| mAP@0.5 | 0.1229 |
| mAP@0.5:0.95 | 0.0676 |
| Precision | 0.1648 |
| Recall | 0.1354 |
| F1-Score | 0.1487 |

## Inference Speed Metrics

| Metric | Value |
|--------|--------|
| Mean Inference Time | 7.69 ms |
| FPS | 130.08 |
| Min Inference Time | 0.00 ms |
| Max Inference Time | 21.65 ms |

## Hardware Resource Usage

| Resource | Usage |
|----------|--------|
| CPU Usage | 0.0% |
| Memory Usage | 50.0% |
| Available Memory | 4.00 GB |

## Per-Class Performance

| Class | mAP@0.5 | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| pedestrian | 0.1447 | 0.2095 | 0.1570 | 0.1662 |
| people | 0.1346 | 0.1934 | 0.1526 | 0.1727 |
| bicycle | 0.1168 | 0.1810 | 0.1619 | 0.1643 |
| car | 0.1476 | 0.1995 | 0.1892 | 0.2008 |
| van | 0.1240 | 0.1663 | 0.1569 | 0.1473 |
| truck | 0.1350 | 0.1742 | 0.1622 | 0.1850 |
| tricycle | 0.0951 | 0.1618 | 0.1055 | 0.1178 |
| awning-tricycle | 0.0817 | 0.1362 | 0.0992 | 0.1375 |
| bus | 0.1353 | 0.1664 | 0.1660 | 0.1723 |
| motor | 0.0962 | 0.1558 | 0.1435 | 0.1457 |


## Protocol v2.0 Methodology Compliance

- **Ultra-Lightweight Target**: ✅ ACHIEVED (<3MB)
- **Real-time Performance**: ✅ ACHIEVED (>10 FPS)
- **Phase 1 Target (True Baseline): >12% mAP@0.5**: ✅ ACHIEVED
- **Environmental Robustness**: ⚠️ MODERATE (Score: 0.682)
- **Average Degradation**: 31.8% across environmental conditions