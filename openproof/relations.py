"""Identity routing and spatially masked Odds redundancy allocation."""
import torch
from torch.nn import functional as F

def reciprocal_centres(destination, source):
    delta = (destination[:, :, None, :2] - source[:, None, :, :2]).abs()
    return (delta <= destination[:, :, None, 2:] * 0.5).all(-1) & (delta <= source[:, None, :, 2:] * 0.5).all(-1)

def routed_identity(query, raw_boxes, refined_boxes, rf_logits, pair_evidence):
    (c, s) = (raw_boxes[..., :2], raw_boxes[..., 2:])
    (lo, hi) = (c - s * 0.5, c + s * 0.5)
    inter = (torch.minimum(hi[:, :, None], hi[:, None]) - torch.maximum(lo[:, :, None], lo[:, None])).clamp_min(0).prod(-1)
    area = s.prod(-1)
    overlap = inter / (area[:, :, None] + area[:, None] - inter).clamp_min(torch.finfo(query.dtype).tiny)
    conditional = rf_logits.softmax(-1)
    affinity = overlap * (conditional @ conditional.transpose(1, 2)) * rf_logits.sigmoid().amax(-1)[:, None, :]
    eye = torch.eye(query.shape[1], device=query.device, dtype=torch.bool)[None]
    supported = reciprocal_centres(refined_boxes.detach(), raw_boxes.detach()) & (pair_evidence.detach() > 0) | eye
    source = affinity.masked_fill(~supported, -torch.inf).argmax(-1)
    return (query.gather(1, source[..., None].expand_as(query)), source)

def relation_odds_probability(pair_logits, rank, boxes, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    q = rank.shape[-1]
    eye = torch.eye(q, device=rank.device, dtype=torch.bool)[None]
    supported = reciprocal_centres(boxes.detach(), boxes.detach()) & ~eye
    delta = (rank[:, None, :].float() - rank[:, :, None].float()) / float(temperature)
    logits = pair_logits.float()
    log_odds = F.logsigmoid(logits) - torch.logaddexp(-delta, F.logsigmoid(-logits))
    log_odds = log_odds.masked_fill(~supported, -torch.inf)
    log_weights = torch.where(eye, torch.zeros_like(log_odds), log_odds)
    denominator = torch.logsumexp(log_weights, -1)
    unique = torch.exp(-denominator).clamp(1e-06, 1.0)
    return (unique, denominator, logits.sigmoid() * supported)
