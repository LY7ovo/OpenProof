"""Previous-model physical target protection under partial annotations."""
from types import MethodType
import torch
FIELDS={"previous_physical_targets","previous_spatial_alignment"}

def install(model, criterion, teacher):
    teacher.eval().requires_grad_(False)
    extract = model.extract_frozen_native_cache
    perceive = model.perceive_cached
    support = model.cached_support_atoms

    @torch.no_grad()
    def frozen(self, images):
        from .boxes import box_cxcywh_to_xyxy
        if self.rf.resolution != teacher.rf.resolution or tuple(self.rf.means) != tuple(teacher.rf.means) or tuple(self.rf.stds) != tuple(teacher.rf.stds):
            raise RuntimeError('Teacher/student pixel preprocessing differs')
        prepared = {}
        preprocess = self.rf.preprocess

        def capture(rows):
            result = preprocess(rows)
            prepared['pixels'] = result
            return result
        self.rf.preprocess = capture
        try:
            current = extract(images)
        finally:
            self.rf.preprocess = preprocess
        old_preprocess = teacher.rf.preprocess
        teacher.rf.preprocess = lambda rows: prepared['pixels']
        try:
            previous = teacher.extract_frozen_native_cache(images)
        finally:
            teacher.rf.preprocess = old_preprocess
        old = teacher.perceive_cached(previous)
        targets = torch.stack((old['query_objectness_logit'].sigmoid(), old['native_candidate_localization_quality_probability'], old['native_candidate_instance_integrity_probability']), -1).detach().float()
        beta = float(teacher.cfg.decoder.quality_ranking_exponent)
        physical = targets[..., 0] * targets[..., 1:].clamp_min(1e-12).prod(-1).pow(beta / 2)
        a = box_cxcywh_to_xyxy(current['box_final']).clamp(0, 1)
        b = box_cxcywh_to_xyxy(previous['box_final']).clamp(0, 1)
        inter = (torch.minimum(a[:, :, None, 2:], b[:, None, :, 2:]) - torch.maximum(a[:, :, None, :2], b[:, None, :, :2])).clamp_min(0).prod(-1)
        aa = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
        bb = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
        overlap = inter / (aa[:, :, None] + bb[:, None, :] - inter).clamp_min(1e-12)
        source = (overlap * physical[:, None]).argmax(-1)
        alignment = overlap.gather(2, source[..., None]).squeeze(-1)
        aligned = targets.gather(1, source[..., None].expand(-1, -1, 3))
        current['previous_physical_targets'] = aligned * alignment[..., None]
        current['previous_spatial_alignment'] = alignment
        return current

    def forward(self, cached):
        out = perceive({k: v for (k, v) in cached.items() if k not in FIELDS})
        if FIELDS.issubset(cached):
            out['_previous_physical_targets'] = cached['previous_physical_targets'].to(self.device_obj, non_blocking=True)
            out['_previous_spatial_alignment'] = cached['previous_spatial_alignment'].to(self.device_obj, non_blocking=True)
        return out

    def atoms(self, cached):
        return support({k: v for (k, v) in cached.items() if k not in FIELDS})
    model.extract_frozen_native_cache = MethodType(frozen, model)
    model.perceive_cached = MethodType(forward, model)
    model.cached_support_atoms = MethodType(atoms, model)
    base = criterion.original
    original_p = base._query_objectness_target_and_class
    original_l = base._native_localization_quality_target
    original_i = base._native_instance_integrity_target
    original_same = base._soft_same_instance_target

    def visible(out, targets):
        if '_authorized_geometry_target' not in out:
            out['_authorized_geometry_target'] = original_l(out, targets)
        if '_authorized_object_target' not in out:
            out['_authorized_object_target'] = original_p(out, targets)[0]
        return (out['_authorized_geometry_target'] > 0) | (out['_authorized_object_target'] >= 1.0)

    def replacement(out, targets, target, index):
        prior = out.get('_previous_physical_targets')
        if prior is None:
            return target
        if prior.shape != (*target.shape, 3):
            raise RuntimeError('Teacher/current candidate shape differs')
        return torch.where(visible(out, targets), target, torch.maximum(target, prior[..., index].detach()))

    def object_target(self, out, targets):
        (value, labels) = original_p(out, targets)
        out['_authorized_object_target'] = value
        return (replacement(out, targets, value, 0), labels)

    def localization(self, out, targets):
        return replacement(out, targets, original_l(out, targets), 1)

    def integrity(self, out, targets):
        return replacement(out, targets, original_i(out, targets), 2)

    def same_target(self, out, targets, localization_target=None, responsibility_cache=None):
        return original_same(out, targets, localization_target=original_l(out, targets), responsibility_cache=responsibility_cache)
    base._query_objectness_target_and_class = MethodType(object_target, base)
    base._native_localization_quality_target = MethodType(localization, base)
    base._native_instance_integrity_target = MethodType(integrity, base)
    base._soft_same_instance_target = MethodType(same_target, base)
    base._physical_target_originals = (original_p, original_l, original_i)
    base._physical_target_original_same = original_same
    return teacher
