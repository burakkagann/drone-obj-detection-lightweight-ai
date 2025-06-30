import torch
import numpy as np
from typing import Tuple, List

class SSDBoxCoder:
    def __init__(self, anchor_boxes: torch.Tensor):
        self.anchor_boxes = anchor_boxes  # Shape: [num_anchors, 4]
        
    def encode(self, boxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert bounding boxes to SSD regression targets.
        
        Args:
            boxes: (tensor) bounding boxes, sized [#obj, 4].
            labels: (tensor) labels, sized [#obj,].
            
        Returns:
            loc_targets: (tensor) encoded bounding boxes, sized [#anchors, 4].
            cls_targets: (tensor) encoded class labels, sized [#anchors,].
        """
        if boxes.numel() == 0:
            return (
                torch.zeros_like(self.anchor_boxes),
                torch.zeros(len(self.anchor_boxes)).long()
            )
            
        ious = box_iou(self.anchor_boxes, boxes)  # [#anchors, #obj]
        max_ious, max_ids = ious.max(1)  # [#anchors,]
        boxes = boxes[max_ids]  # [#anchors, 4]
        
        loc_xy = (boxes[:, :2] - self.anchor_boxes[:, :2]) / self.anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / self.anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        
        cls_targets = labels[max_ids]
        cls_targets[max_ious < 0.5] = 0  # background
        return loc_targets, cls_targets
        
    def decode(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor,
              score_thresh: float = 0.6, nms_thresh: float = 0.45) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Decode predicted loc/cls back to real box locations and class labels.
        
        Args:
            loc_preds: (tensor) predicted loc, sized [batch, #anchors, 4].
            cls_preds: (tensor) predicted conf, sized [batch, #anchors, #classes].
            score_thresh: (float) threshold for object confidence score.
            nms_thresh: (float) threshold for box nms.
            
        Returns:
            boxes: (list) [[x1, y1, x2, y2], ...], normalized.
            labels: (list) class labels.
            scores: (list) score values.
        """
        xy = loc_preds[:, :2] * self.anchor_boxes[:, 2:] + self.anchor_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:]) * self.anchor_boxes[:, 2:]
        box_preds = torch.cat([xy - wh/2, xy + wh/2], 1)
        
        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        
        for i in range(num_classes-1):
            score = cls_preds[:, i+1]  # class i+1
            mask = score > score_thresh
            if not mask.any():
                continue
                
            box = box_preds[mask]
            score = score[mask]
            
            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor([i+1]).expand_as(keep))
            scores.append(score[keep])
            
        if len(boxes) == 0:
            return [], [], []
            
        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute the intersection over union of two sets of boxes.
    
    Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [M,4].
        
    Return:
        (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)
    
    lt = torch.max(
        box1[:, None, :2],  # [N,1,2]
        box2[:, :2]  # [M,2]
    )  # [N,M,2]
    
    rb = torch.min(
        box1[:, None, 2:],  # [N,1,2]
        box2[:, 2:]  # [M,2]
    )  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def box_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Non maximum suppression.
    
    Args:
        boxes: (tensor) bounding boxes, sized [N,4].
        scores: (tensor) confidence scores, sized [N,].
        threshold: (float) overlap threshold.
        
    Return:
        keep: (tensor) selected indices.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0]
        keep.append(i)
        
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
        
    return torch.LongTensor(keep) 