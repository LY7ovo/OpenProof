"""Parameter-free geometry-aware encoder query selection for OWOD."""
from __future__ import annotations
import torch

def priorities(logits, boxes, valid):
    score = logits.float().amax(-1).masked_fill(~valid, -torch.inf)
    order = score.argsort(dim=1, descending=True, stable=True)
    ranked = boxes.float().gather(1, order[..., None].expand(-1, -1, 4))
    (lo, hi) = (ranked[..., :2], ranked[..., 2:])
    area = (hi - lo).clamp_min(0).prod(-1)
    overlap = (torch.minimum(hi[:, :, None], hi[:, None, :]) - torch.maximum(lo[:, :, None], lo[:, None, :])).clamp_min(0).prod(-1)
    overlap = overlap / (area[:, :, None] + area[:, None, :] - overlap).clamp_min(1e-12)
    higher = torch.ones_like(overlap, dtype=torch.bool).tril(-1)
    ranked_valid = valid.gather(1, order)
    overlap.masked_fill_(~(higher & ranked_valid[:, :, None] & ranked_valid[:, None, :]), 0)
    redundancy = overlap.amax(-1)
    priority = score.gather(1, order).sigmoid() * (1 - redundancy)
    priority.masked_fill_(~ranked_valid, -torch.inf)
    return torch.empty_like(priority).scatter_(1, order, priority)

class Selection:

    def __init__(self, core):
        (self.core, self.enabled, self.audit) = (core, True, None)
        tf = core.transformer
        if tf.num_queries != 300 or not tf.two_stage or getattr(tf, 'num_registers', 0):
            raise RuntimeError('This selector requires the checked 300-query two-stage detector')
        (self.pool, self.proposals) = (None, None)
        self.handles = [tf.register_forward_pre_hook(self.before, with_kwargs=True), tf.enc_out_class_embed[0].register_forward_hook(self.classifier)]

    def before(self, module, args, kwargs):
        if not self.enabled:
            return
        if module.training or any((p.requires_grad for p in self.core.parameters())):
            raise RuntimeError('Geometry preselection is restricted to the frozen evaluation RF')
        from rfdetr.models.transformer import gen_encoder_output_proposals
        srcs = args[0] if args else kwargs['srcs']
        masks = args[1] if len(args) > 1 else kwargs.get('masks')
        memory = torch.cat([s.flatten(2).transpose(1, 2) for s in srcs], 1)
        mask = None if masks is None else torch.cat([m.flatten(1) for m in masks], 1)
        shapes = [(s.shape[-2], s.shape[-1]) for s in srcs]
        (_, self.proposals) = gen_encoder_output_proposals(memory, mask, shapes, unsigmoid=not module.bbox_reparam)
        self.pool = memory.shape[1]

    def classifier(self, module, args, output):
        if not self.enabled or output.shape[1] != self.pool:
            return output
        tf = self.core.transformer
        delta = tf.enc_out_bbox_embed[0](args[0])
        if tf.bbox_reparam:
            boxes = torch.cat((delta[..., :2] * self.proposals[..., 2:] + self.proposals[..., :2], delta[..., 2:].exp() * self.proposals[..., 2:]), -1)
        else:
            boxes = (delta + self.proposals).sigmoid()
        valid = torch.isfinite(self.proposals).all(-1) & torch.isfinite(boxes).all(-1)
        xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1).clamp(0, 1)
        xyxy = torch.nan_to_num(xyxy)
        priority = priorities(output, xyxy, valid)
        selected = priority.topk(300, dim=1).indices
        self.audit = dict(original=output.amax(-1).topk(300, dim=1).indices.detach(), selected=selected.detach(), pool=self.pool)
        return priority[..., None]

def install(core):
    if not hasattr(core, '_geometry_selection'):
        core._geometry_selection = Selection(core)
    return core._geometry_selection
