"""Frozen RF-DETR Small and exact decoder/cross-attention capture."""
from __future__ import annotations
import torch
from torch import nn
from torchvision.transforms import functional as TF

class FrozenRF(nn.Module):

    def __init__(self, checkpoint, classes, device='cuda', geometry=False):
        super().__init__()
        from rfdetr import RFDETRSmall
        from types import SimpleNamespace
        self.device_obj = torch.device(device)
        self.checkpoint=__import__('pathlib').Path(checkpoint).resolve()
        self.geometry=bool(geometry)
        self.rf_cfg = SimpleNamespace(core_amp=False)
        self.contract = SimpleNamespace(channels=classes + 1)
        api = RFDETRSmall(pretrain_weights=str(checkpoint), num_classes=classes, device='cpu', resolution=512)
        self.core = api.model.model
        self.decoder = self.core.transformer.decoder
        self.resolution = 512
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.core.eval().requires_grad_(False).to(self.device_obj)
        self._transformer_capture = None
        self._hook = self.core.transformer.register_forward_hook(self._capture_transformer)
        if geometry:
            from .selection import install
            install(self.core)

    def _capture_transformer(self, module, args, out):
        self._transformer_capture = out

    def train(self, mode=True):
        super().train(False)
        self.core.eval()
        return self

    @staticmethod
    def _detach_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, tuple):
            return tuple((FrozenRF._detach_tree(x) for x in value))
        if isinstance(value, list):
            return [FrozenRF._detach_tree(x) for x in value]
        if isinstance(value, dict):
            return {k: FrozenRF._detach_tree(v) for (k, v) in value.items()}
        return value

    def _aligned_decoder_references(self, hs: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        if references.ndim == 3:
            references = references.unsqueeze(0)
        if references.ndim != 4 or references.shape[1:3] != hs.shape[1:3] or references.shape[-1] != 4:
            raise RuntimeError(f'RF reference trajectory shape invalid: hs={tuple(hs.shape)} refs={tuple(references.shape)}')
        if int(references.shape[0]) == 1 and int(hs.shape[0]) > 1:
            if not bool(getattr(self.core, 'lite_refpoint_refine', False)):
                raise RuntimeError('RF exposed a singleton decoder reference for multiple hidden layers without lite_refpoint_refine=True')
            references = references.expand(hs.shape[0], -1, -1, -1)
        elif int(references.shape[0]) != int(hs.shape[0]):
            raise RuntimeError(f'RF reference trajectory length mismatch: hidden_layers={hs.shape[0]} reference_layers={references.shape[0]}')
        return references.float()

    def _normalized_decoder_references(self, hs: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        aligned = FrozenRF._aligned_decoder_references(self, hs, references)
        if bool(getattr(self.core, 'bbox_reparam', False)):
            return aligned
        return aligned.sigmoid()

    def _decoder_box_trajectory(self, hs: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        references = FrozenRF._aligned_decoder_references(self, hs, references)
        head = getattr(self.core, 'bbox_embed', None)
        if head is None:
            raise RuntimeError('RF core has no bbox_embed needed for per-layer box trajectory')
        if isinstance(head, nn.ModuleList):
            if len(head) < hs.shape[0]:
                raise RuntimeError('RF bbox_embed ModuleList is shorter than decoder trajectory')
            delta = torch.stack([head[i](hs[i]) for i in range(hs.shape[0])], dim=0)
        else:
            delta = head(hs)
        if bool(getattr(self.core, 'bbox_reparam', False)):
            cxcy = delta[..., :2] * references[..., 2:] + references[..., :2]
            wh = delta[..., 2:].exp() * references[..., 2:]
            boxes = torch.cat([cxcy, wh], dim=-1)
        else:
            boxes = (delta + references).sigmoid()
        return boxes.float()

    def _decoder_class_trajectory(self, hs: torch.Tensor) -> torch.Tensor:
        head = getattr(self.core, 'class_embed', None)
        if head is None:
            raise RuntimeError('RF core has no class_embed needed for cross-layer Known consensus')
        if isinstance(head, nn.ModuleList):
            if len(head) < hs.shape[0]:
                raise RuntimeError('RF class_embed ModuleList is shorter than decoder trajectory')
            logits = torch.stack([head[i](hs[i]) for i in range(hs.shape[0])], dim=0)
        else:
            logits = head(hs)
        return logits.float()

    def preprocess(self, images):
        rows = []
        for image in images:
            if isinstance(image, torch.Tensor):
                t = image.float()
                if t.ndim != 3:
                    raise ValueError('tensor image must be CHW')
                if t.max() > 1.5:
                    t = t / 255.0
            else:
                t = TF.to_tensor(image.convert('RGB'))
            t = TF.resize(t, [self.resolution, self.resolution], antialias=False)
            t = TF.normalize(t, self.means, self.stds)
            rows.append(t)
        return torch.stack(rows).to(self.device_obj, non_blocking=True)

    def _capture_perception(self, out, cross_records=None):
        if not isinstance(out, dict) or 'pred_logits' not in out or 'pred_boxes' not in out:
            raise RuntimeError('RF forward output contract violated')
        cap = self._transformer_capture
        if cap is None:
            raise RuntimeError('decoder trajectory hook failed')
        hs = cap[0] if isinstance(cap, tuple) else cap
        refs = cap[1] if isinstance(cap, tuple) and len(cap) > 1 and isinstance(cap[1], torch.Tensor) else None
        if hs.ndim == 3:
            hs = hs.unsqueeze(0)
        if hs.ndim != 4:
            raise RuntimeError(f'decoder hidden trajectory shape invalid: {tuple(hs.shape)}')
        logits = out['pred_logits'].float()
        boxes = out['pred_boxes'].float()
        if logits.shape[-1] != self.contract.channels:
            raise RuntimeError('RF classifier channel count changed')
        if hs.shape[1] != logits.shape[0] or hs.shape[2] != logits.shape[1]:
            raise RuntimeError('RF hidden/query shape mismatch')
        if refs is None:
            raise RuntimeError('RF transformer did not expose decoder reference trajectory')
        layer_boxes = self._decoder_box_trajectory(hs.float(), refs.float())
        layer_class_logits = self._decoder_class_trajectory(hs.float())
        layer_references = self._normalized_decoder_references(hs.float(), refs.float())
        if layer_boxes.shape[-3:] != (logits.shape[0], logits.shape[1], 4):
            raise RuntimeError('RF per-layer box trajectory does not align with native queries')
        if not torch.allclose(layer_boxes[-1], boxes, atol=2e-05, rtol=2e-05):
            diff = float((layer_boxes[-1] - boxes).abs().max().detach().cpu())
            raise RuntimeError(f'RF reconstructed final decoder box is not native-identical (max_abs={diff})')
        if tuple(layer_class_logits.shape[-3:]) != tuple(logits.shape):
            raise RuntimeError(f'RF per-layer classifier trajectory does not align with native logits: {tuple(layer_class_logits.shape)} vs {tuple(logits.shape)}')
        if not torch.allclose(layer_class_logits[-1], logits, atol=2e-05, rtol=2e-05):
            diff = float((layer_class_logits[-1] - logits).abs().max().detach().cpu())
            raise RuntimeError(f'RF reconstructed final decoder classifier is not native-identical (max_abs={diff})')
        result = {'pred_logits': logits, 'pred_boxes': boxes, 'decoder_hidden_layers': hs.float(), 'decoder_box_layers': layer_boxes, 'decoder_class_logit_layers': layer_class_logits, 'decoder_reference_layers': layer_references}
        if cross_records is not None:
            if len(cross_records) != hs.shape[0] or any((r.get('query') is None or r.get('output') is None for r in cross_records)):
                raise RuntimeError('RF decoder cross-attention capture is incomplete')
            result['decoder_cross_query_layers'] = torch.stack([r['query'].float() for r in cross_records], dim=0)
            result['decoder_cross_context_layers'] = torch.stack([r['output'].float() for r in cross_records], dim=0)
            first_tail = list(cross_records[0].get('tail_args', ()))
            if len(first_tail) < 2 or not isinstance(first_tail[1], torch.Tensor):
                raise RuntimeError('RF cross-attention capture did not expose encoder image memory')
            encoder_memory = first_tail[1]
            if encoder_memory.ndim != 3 or encoder_memory.shape[0] != logits.shape[0] or encoder_memory.shape[-1] != hs.shape[-1]:
                raise RuntimeError(f'RF encoder memory shape invalid for open-world transformer: {tuple(encoder_memory.shape)}')
            result['encoder_memory_tokens'] = encoder_memory.float()
            encoder_mask = None
            if len(first_tail) > 4 and isinstance(first_tail[4], torch.Tensor):
                candidate = first_tail[4]
                if candidate.ndim == 2 and candidate.shape[0] == logits.shape[0] and (candidate.shape[1] == encoder_memory.shape[1]):
                    encoder_mask = candidate.bool()
            result['encoder_memory_key_padding_mask'] = encoder_mask
            result['_cross_attention_records'] = cross_records
        return result

    def _run_core(self, batch, capture_cross_attention=False):
        self._transformer_capture = None
        handles = []
        cross_records = None
        layers = getattr(self.decoder, 'layers', None)
        if capture_cross_attention:
            if layers is None:
                raise RuntimeError('RF decoder has no layer list for Surprise Query capture')
            cross_records = [{} for _ in range(len(layers))]
            for (layer_id, layer) in enumerate(layers):
                cross = getattr(layer, 'cross_attn', None)
                if not isinstance(cross, nn.Module):
                    raise RuntimeError(f'RF decoder layer {layer_id} has no cross_attn module')
                record = cross_records[layer_id]

                def pre_hook(_m, args, kwargs, _record=record):
                    if not args or not isinstance(args[0], torch.Tensor):
                        raise RuntimeError('RF cross-attention query capture failed')
                    _record['query'] = args[0].detach()
                    _record['tail_args'] = tuple((self._detach_tree(x) for x in args[1:]))
                    _record['kwargs'] = {k: self._detach_tree(v) for (k, v) in kwargs.items()}

                def post_hook(_m, args, kwargs, out, _record=record):
                    value = out[0] if isinstance(out, tuple) else out
                    if not isinstance(value, torch.Tensor):
                        raise RuntimeError('RF cross-attention output capture failed')
                    _record['output'] = value.detach()
                try:
                    handles.append(cross.register_forward_pre_hook(pre_hook, with_kwargs=True))
                    handles.append(cross.register_forward_hook(post_hook, with_kwargs=True))
                except TypeError as exc:
                    raise RuntimeError('installed PyTorch lacks with_kwargs hooks required for exact Surprise Query replay') from exc
        amp = bool(self.rf_cfg.core_amp and self.device_obj.type == 'cuda')
        try:
            with torch.amp.autocast(device_type=self.device_obj.type, enabled=amp):
                out = self.core(batch)
        finally:
            for handle in handles:
                handle.remove()
        return self._capture_perception(out, cross_records)

    @torch.no_grad()
    def native_pass(self, images, capture_cross_attention=True):
        batch = self.preprocess(images)
        p = self._run_core(batch, capture_cross_attention=capture_cross_attention)
        result = {'native_logits': p['pred_logits'].detach(), 'pred_boxes': p['pred_boxes'].detach(), 'decoder_hidden_layers': p['decoder_hidden_layers'].detach(), 'decoder_box_layers': p['decoder_box_layers'].detach(), 'decoder_class_logit_layers': p['decoder_class_logit_layers'].detach(), 'decoder_reference_layers': p['decoder_reference_layers'].detach(), '_preprocessed_batch': batch.detach()}
        if capture_cross_attention:
            result['decoder_cross_query_layers'] = p['decoder_cross_query_layers'].detach()
            result['decoder_cross_context_layers'] = p['decoder_cross_context_layers'].detach()
            result['encoder_memory_tokens'] = p['encoder_memory_tokens'].detach()
            mask = p.get('encoder_memory_key_padding_mask')
            result['encoder_memory_key_padding_mask'] = None if mask is None else mask.detach()
            result['_cross_attention_records'] = p['_cross_attention_records']
        return result
