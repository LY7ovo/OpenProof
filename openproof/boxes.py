from __future__ import annotations
import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    (cx, cy, w, h) = x.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    (x0, y0, x1, y1) = x.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    if torch.any(union < -1e-06):
        raise RuntimeError(f'negative IoU union detected: min={float(union.min())}')
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    if torch.any(iou > 1.00001) or torch.any(iou < -1e-06) or (not torch.isfinite(iou).all()):
        raise RuntimeError('invalid IoU detected')
    return (iou, union)

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    (iou, union) = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area = box_area(torch.cat([lt, rb], dim=-1))
    giou = iou - torch.where(area > 0, (area - union) / area, torch.zeros_like(area))
    if torch.any(giou > 1.00001) or not torch.isfinite(giou).all():
        raise RuntimeError('invalid GIoU detected')
    return giou
