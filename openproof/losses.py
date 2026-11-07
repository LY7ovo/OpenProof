"""The seven retained mainline objectives, with unchanged targets and reductions."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .boxes import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from .model import make_roots, stack_root_field
SEVEN_KEYS=('loss_query_objectness', 'loss_r11p4_hardtail_relative_object_ranking', 'loss_localization_quality', 'loss_instance_integrity', 'loss_soft_same_instance', 'loss_joint_quality_ranking', 'loss_formal_continuous_related_quality_ranking')

class Criterion:

    def __init__(self, cfg):
        self.original=self
        self.cfg = cfg
        self._class_image_counts = None
        self._class_balance_weights = None

    def set_class_balance_counts(self, image_counts) -> None:
        counts = torch.as_tensor(image_counts, dtype=torch.float32).reshape(-1)
        expected = int(self.cfg.num_known_classes)
        if counts.numel() != expected:
            raise ValueError(f'class balance count length {counts.numel()} != {expected}')
        if bool((counts <= 0).any()):
            raise ValueError(f'every installed Base class needs positive training-image support: {counts.tolist()}')
        inv = counts.reciprocal()
        weights = inv / inv.mean().clamp_min(1e-12)
        self._class_image_counts = counts.clone()
        self._class_balance_weights = weights.clone()

    def _class_weights(self, device) -> torch.Tensor:
        if self._class_balance_weights is None:
            return torch.ones(int(self.cfg.num_known_classes), device=device, dtype=torch.float32)
        return self._class_balance_weights.to(device=device, dtype=torch.float32)

    def _macro_class_loss(self, per_object: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if per_object.numel() == 0:
            return per_object.sum() * 0.0
        weights = self._class_weights(per_object.device)
        total = per_object.sum() * 0.0
        for c in range(int(self.cfg.num_known_classes)):
            mask = labels == c
            if bool(mask.any()):
                total = total + weights[c] * per_object[mask].mean()
        return total / float(self.cfg.num_known_classes)

    @torch.no_grad()
    def _match_native_candidates(self, out, targets):
        boxes = out['native_candidate_boxes'].float()
        matches = []
        for (b, target) in enumerate(targets):
            gt = target['boxes'].to(boxes.device).float()
            if gt.numel() == 0:
                matches.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            l1 = torch.cdist(boxes[b], gt, p=1)
            giou = generalized_box_iou(box_cxcywh_to_xyxy(boxes[b]), box_cxcywh_to_xyxy(gt))
            cost = (float(self.cfg.loss.matcher_bbox_cost) * l1 + float(self.cfg.loss.matcher_giou_cost) * (1.0 - giou)).double()
            (qi, gi) = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(gi, dtype=torch.long)))
        return matches

    def _physical_roots(self, out):
        return make_roots(out)

    def _match_physical_roots(self, roots, targets):
        matches = []
        for (b, target) in enumerate(targets):
            gt = target['boxes'].float()
            if gt.numel() == 0 or not roots[b]:
                matches.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            rboxes = stack_root_field(roots[b], 'box').float()
            gt = gt.to(rboxes.device)
            l1 = torch.cdist(rboxes, gt, p=1)
            giou = generalized_box_iou(box_cxcywh_to_xyxy(rboxes), box_cxcywh_to_xyxy(gt))
            cost = (float(self.cfg.loss.matcher_bbox_cost) * l1 + float(self.cfg.loss.matcher_giou_cost) * (1.0 - giou)).double()
            (ri, gi) = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((torch.as_tensor(ri, dtype=torch.long), torch.as_tensor(gi, dtype=torch.long)))
        return matches

    @staticmethod
    def _proper_quality_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.float(), target.detach().float().clamp(0.0, 1.0))

    @torch.no_grad()
    def _native_localization_quality_target(self, out, targets) -> torch.Tensor:
        logits = out['native_candidate_localization_quality_logit']
        target = torch.zeros_like(logits, dtype=torch.float32)
        for (b, item) in enumerate(targets):
            gt = item['boxes'].to(logits.device).float()
            if gt.numel() == 0:
                continue
            native = out['native_candidate_boxes'][b].float().detach()
            (iou, _) = box_iou(box_cxcywh_to_xyxy(native), box_cxcywh_to_xyxy(gt))
            target[b] = iou.amax(dim=1).clamp(0.0, 1.0)
        return target

    @torch.no_grad()
    def _native_instance_integrity_target(self, out, targets) -> torch.Tensor:
        logits = out['native_candidate_instance_integrity_logit']
        target = torch.zeros_like(logits, dtype=torch.float32)
        eps = 1e-06
        for (b, item) in enumerate(targets):
            gt = item['boxes'].to(logits.device).float()
            if gt.numel() == 0:
                continue
            pred_xy = box_cxcywh_to_xyxy(out['native_candidate_boxes'][b].float().detach())
            gt_xy = box_cxcywh_to_xyxy(gt)
            (iou, _) = box_iou(pred_xy, gt_xy)
            best_idx = iou.argmax(dim=1)
            chosen = gt_xy.index_select(0, best_idx)
            lt = torch.maximum(pred_xy[:, :2], chosen[:, :2])
            rb = torch.minimum(pred_xy[:, 2:], chosen[:, 2:])
            wh = (rb - lt).clamp_min(0.0)
            best_intersection = wh[:, 0] * wh[:, 1]
            gt_area = (chosen[:, 2] - chosen[:, 0]).clamp_min(0.0) * (chosen[:, 3] - chosen[:, 1]).clamp_min(0.0)
            target[b] = (best_intersection / gt_area.clamp_min(eps)).clamp(0.0, 1.0)
        return target

    @staticmethod
    @torch.no_grad()
    def _soft_gt_responsibility(native_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        native = native_boxes.float().detach()
        gt = gt_boxes.to(native.device).float()
        q = int(native.shape[0])
        if gt.numel() == 0:
            empty = native.new_zeros((q, 0), dtype=torch.float32)
            scene = native.new_ones((q,), dtype=torch.float32)
            affinity = empty.clone()
            return (empty, scene, affinity)
        native_xy = box_cxcywh_to_xyxy(native)
        gt_xy = box_cxcywh_to_xyxy(gt)
        (iou, _) = box_iou(native_xy, gt_xy)
        center_err = ((native[:, None, :2] - gt[None, :, :2]).abs() / gt[None, :, 2:].clamp_min(0.0001)).mean(-1)
        scale_err = (native[:, None, 2:].clamp_min(0.0001).log() - gt[None, :, 2:].clamp_min(0.0001).log()).abs().mean(-1)
        geometry_consistency = torch.exp(-(center_err + 0.5 * scale_err)).clamp(0.0, 1.0)
        affinity = (iou.clamp(0.0, 1.0) * torch.sqrt(geometry_consistency.clamp_min(1e-08))).clamp(0.0, 1.0)
        scene_mass = torch.prod((1.0 - affinity).clamp(0.0, 1.0), dim=-1)
        denominator = (affinity.sum(dim=-1) + scene_mass).clamp_min(1e-08)
        gt_responsibility = affinity / denominator[:, None]
        scene_responsibility = scene_mass / denominator
        return (gt_responsibility, scene_responsibility, affinity)

    @torch.no_grad()
    def _build_native_responsibility_cache(self, out, targets, box_key='native_candidate_boxes'):
        boxes = out[box_key]
        cache = []
        for (b, item) in enumerate(targets):
            gt = item['boxes'].to(boxes.device).float()
            cache.append(self._soft_gt_responsibility(boxes[b].float().detach(), gt))
        return cache

    @torch.no_grad()
    def _soft_same_instance_target(self, out, targets, localization_target: torch.Tensor | None=None, responsibility_cache=None) -> tuple[torch.Tensor, torch.Tensor]:
        del localization_target
        logits = out.get('native_candidate_soft_same_instance_evidence_logits', out['native_candidate_soft_same_instance_logits'])
        target = torch.zeros_like(logits, dtype=torch.float32)
        weight = torch.ones_like(logits, dtype=torch.float32)
        for (b, item) in enumerate(targets):
            gt = item['boxes'].to(logits.device).float()
            if responsibility_cache is None:
                native = out['native_candidate_boxes'][b].float().detach()
                (gt_responsibility, _, _) = self._soft_gt_responsibility(native, gt)
            else:
                gt_responsibility = responsibility_cache[b][0]
            if gt_responsibility.shape[-1] > 0:
                target[b] = gt_responsibility @ gt_responsibility.transpose(0, 1)
        eye = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)[None]
        target = target.clamp(0.0, 1.0).masked_fill(eye, 0.0)
        weight = weight.masked_fill(eye, 0.0)
        return (target, weight)

    @staticmethod
    def _balanced_pair_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        q = logits.shape[-1]
        if q <= 1:
            return logits.sum() * 0.0
        tri = torch.triu(torch.ones((q, q), device=logits.device, dtype=torch.bool), diagonal=1)[None]
        valid_weight = weight.float() * tri.float()
        target = target.float().clamp(0.0, 1.0)
        pos_weight = valid_weight * target
        neg_weight = valid_weight * (1.0 - target)
        pos_mass = pos_weight.sum()
        neg_mass = neg_weight.sum()
        pos_term = (pos_weight * F.softplus(-logits.float())).sum() / pos_mass.clamp_min(1e-06)
        neg_term = (neg_weight * F.softplus(logits.float())).sum() / neg_mass.clamp_min(1e-06)
        pos_active = (pos_mass > 1e-08).to(logits.dtype).detach()
        neg_active = (neg_mass > 1e-08).to(logits.dtype).detach()
        return (pos_active * pos_term + neg_active * neg_term) / (pos_active + neg_active).clamp_min(1.0)

    @torch.no_grad()
    def _query_objectness_target_and_class(self, out, targets) -> tuple[torch.Tensor, torch.Tensor]:
        logits = out['query_objectness_logit']
        target = torch.zeros_like(logits, dtype=torch.float32)
        target_class = torch.full_like(logits, -1, dtype=torch.long)
        for (b, item) in enumerate(targets):
            gt = item['boxes'].to(logits.device).float()
            if gt.numel() == 0:
                continue
            labels = item['labels'].to(logits.device).long()
            native_source = out['native_candidate_boxes'] if 'native_candidate_boxes' in out else out['native_pred_boxes']
            native = native_source[b].float().detach()
            l1 = torch.cdist(native, gt, p=1)
            giou = generalized_box_iou(box_cxcywh_to_xyxy(native), box_cxcywh_to_xyxy(gt))
            cost = l1 + (1.0 - giou)
            affinity = torch.exp(-cost).clamp(0.0, 1.0)
            target[b] = affinity.amax(dim=1)
            best_gt = affinity.argmax(dim=1)
            target_class[b] = labels.index_select(0, best_gt)
            (q_idx, g_idx) = linear_sum_assignment(cost.detach().cpu().numpy())
            q_idx = torch.as_tensor(q_idx, device=logits.device, dtype=torch.long)
            g_idx = torch.as_tensor(g_idx, device=logits.device, dtype=torch.long)
            if q_idx.numel():
                target[b, q_idx] = 1.0
                target_class[b, q_idx] = labels.index_select(0, g_idx)
        return (target, target_class)

    def _r11p4_hardtail_relative_object_ranking_loss(self, logits: torch.Tensor, target: torch.Tensor, native_matches) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = logits.float()
        target = target.float().clamp(0.0, 1.0)
        risk_parts = []
        valid_parts = []
        gap_parts = []
        mass_parts = []
        q = logits.shape[1]
        for (b, (query_cpu, _gt_cpu)) in enumerate(native_matches):
            pos = query_cpu.to(logits.device, non_blocking=True).long()
            if pos.numel() == 0:
                continue
            pos_mask = torch.zeros(q, device=logits.device, dtype=torch.bool)
            pos_mask.scatter_(0, pos, True)
            bg_weight = (1.0 - target[b]).masked_fill(pos_mask, 0.0)
            mass = bg_weight.sum()
            logw = torch.log(bg_weight.clamp_min(1e-12))
            pos_logits = logits[b].index_select(0, pos)
            pair = F.softplus(logits[b].unsqueeze(0) - pos_logits.unsqueeze(1))
            per_gt = torch.logsumexp(logw.unsqueeze(0) + pair, dim=1) - torch.log(mass.clamp_min(1e-06))
            valid = (mass > 1e-08).expand_as(per_gt)
            risk_parts.append(per_gt)
            valid_parts.append(valid)
            mean_bg = (bg_weight * logits[b].detach()).sum() / mass.clamp_min(1e-06)
            gap_parts.append(pos_logits.detach() - mean_bg)
            mass_parts.append(mass.detach().expand_as(per_gt))
        zero = logits.sum() * 0.0
        if not risk_parts:
            return (zero, {'instance_count': torch.tensor(0, device=logits.device), 'mean_gap': zero.detach(), 'background_mass': zero.detach(), 'hard_risk_mean': zero.detach(), 'hard_risk_max': zero.detach()})
        risks = torch.cat(risk_parts)
        valid = torch.cat(valid_parts)
        risks = risks[valid]
        if risks.numel() == 0:
            return (zero, {'instance_count': torch.tensor(0, device=logits.device), 'mean_gap': zero.detach(), 'background_mass': zero.detach(), 'hard_risk_mean': zero.detach(), 'hard_risk_max': zero.detach()})
        loss = torch.logsumexp(risks, dim=0) - torch.log(risks.new_tensor(float(risks.numel())))
        gaps = torch.cat(gap_parts)[torch.cat(valid_parts)]
        masses = torch.cat(mass_parts)[torch.cat(valid_parts)]
        return (loss, {'instance_count': torch.tensor(risks.numel(), device=logits.device), 'mean_gap': gaps.mean(), 'background_mass': masses.mean(), 'hard_risk_mean': risks.detach().mean(), 'hard_risk_max': risks.detach().amax()})

    def _joint_quality_ranking_loss(self, out, targets, native_matches, objectness_target, localization_target, integrity_target, same_instance_target=None):
        from .boxes import box_cxcywh_to_xyxy, box_iou
        unique_target = torch.ones_like(objectness_target, dtype=torch.float32)
        beta = float(getattr(getattr(self.cfg, 'decoder', None), 'quality_ranking_exponent', 0.5))
        target_quality = torch.sqrt((localization_target * integrity_target).clamp_min(1e-12)).pow(beta)
        quality_target = (objectness_target * target_quality).detach().clamp(0.0, 1.0)
        score = torch.log(out['native_candidate_existence_probability'].float().clamp_min(1e-08)) + 0.5 * beta * torch.log(out['native_candidate_localization_quality_probability'].float().clamp_min(1e-08)) + 0.5 * beta * torch.log(out['native_candidate_instance_integrity_probability'].float().clamp_min(1e-08))
        k = max(1, int(getattr(self.cfg.loss, 'quality_ranking_hard_negatives', 8)))
        (per_positive, labels, gap_values, neg_target_values) = ([], [], [], [])
        pair_count = same_gt_pairs = fallback_pairs = positive_count = 0
        for (b, (query_cpu, gt_cpu)) in enumerate(native_matches):
            if query_cpu.numel() == 0:
                continue
            qpos = query_cpu.to(score.device)
            gti = gt_cpu.to(score.device)
            matched_mask = torch.zeros(score.shape[1], device=score.device, dtype=torch.bool)
            matched_mask[qpos] = True
            gt_labels = targets[b]['labels'].to(score.device).long()
            with torch.no_grad():
                boxes = out['native_candidate_boxes'][b].float().detach()
                gt = targets[b]['boxes'].to(score.device).float()
                (ious, _) = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
                nearest_gt = ious.argmax(dim=1)
            for j in range(int(qpos.numel())):
                (qi, gi) = (qpos[j], gti[j])
                pos_target = quality_target[b, qi]
                eligible = ~matched_mask & (quality_target[b] < pos_target)
                is_localization_pair = False
                if bool(ious[qi, gi] >= 0.5):
                    overlap = ious[:, gi]
                    local = eligible & (nearest_gt == gi) & (overlap >= 0.1) & (overlap < 0.5)
                    if bool(local.any()):
                        eligible = local
                        is_localization_pair = True
                idx = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                if idx.numel() > k:
                    top = torch.topk(score[b].detach().index_select(0, idx), k=k, largest=True).indices
                    idx = idx.index_select(0, top)
                neg_score = score[b].index_select(0, idx)
                neg_target = quality_target[b].index_select(0, idx)
                weight = (pos_target - neg_target).clamp_min(0.0)
                denom = weight.sum()
                if not bool(denom > 1e-08):
                    continue
                pair = F.softplus(-(score[b, qi] - neg_score))
                per_positive.append((weight * pair).sum() / denom.clamp_min(1e-06))
                labels.append(gt_labels[gi])
                gap_values.append((score[b, qi].detach() - neg_score.detach()).mean())
                neg_target_values.append(neg_target.mean())
                pair_count += int(idx.numel())
                positive_count += 1
                if is_localization_pair:
                    same_gt_pairs += int(idx.numel())
                else:
                    fallback_pairs += int(idx.numel())
        self._same_gt_joint_diag = dict(pair_count=pair_count, same_gt_pairs=same_gt_pairs, fallback_pairs=fallback_pairs, positive_count=positive_count)
        zero = score.sum() * 0.0
        if not per_positive:
            return (zero, dict(target=quality_target, uniqueness_target=unique_target, pair_gap=zero.detach(), negative_target=zero.detach(), pair_count=torch.tensor(0, device=score.device)))
        loss = self._macro_class_loss(torch.stack(per_positive), torch.stack(labels).long())
        return (loss, dict(target=quality_target, uniqueness_target=unique_target, pair_gap=torch.stack(gap_values).mean().detach(), negative_target=torch.stack(neg_target_values).mean().detach(), pair_count=torch.tensor(pair_count, device=score.device)))

    def _formal_continuous_ranking_losses(self, out, roots, root_matches, targets, localization_target: torch.Tensor, integrity_target: torch.Tensor, same_instance_target: torch.Tensor, responsibility_cache=None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        device = out['native_candidate_boxes'].device
        zero = out['native_candidate_object_logit'].sum() * 0.0
        family_terms: list[torch.Tensor] = []
        family_labels: list[torch.Tensor] = []
        routed_terms: list[torch.Tensor] = []
        routed_labels: list[torch.Tensor] = []
        family_gaps: list[torch.Tensor] = []
        routed_gaps: list[torch.Tensor] = []
        hard_unknown_scores: list[torch.Tensor] = []
        relatedness_values: list[torch.Tensor] = []
        loc_weight_values: list[torch.Tensor] = []
        integrity_weight_values: list[torch.Tensor] = []
        duplicate_weight_values: list[torch.Tensor] = []
        pair_count = 0
        k = max(1, int(getattr(self.cfg.loss, 'formal_continuous_hard_negatives', 8)))
        family_margin = float(getattr(self.cfg.loss, 'formal_continuous_known_unknown_margin_nats', 0.5))
        quality_margin = float(getattr(self.cfg.loss, 'formal_continuous_quality_margin_nats', 0.2))
        eps = 1e-08
        for (b, (root_cpu, gt_cpu)) in enumerate(root_matches):
            image_roots = roots[b]
            if not image_roots or root_cpu.numel() == 0:
                continue
            root_idx = root_cpu.to(device)
            gt_idx = gt_cpu.to(device)
            gt_boxes = targets[b]['boxes'].to(device).float()
            gt_labels = targets[b]['labels'].to(device).long()
            if gt_boxes.numel() == 0:
                continue
            representatives = stack_root_field(image_roots, 'representative').long().to(device)
            ranking_quality = stack_root_field(image_roots, 'ranking_object_probability').float()
            unknown_score = stack_root_field(image_roots, 'unknown_ranking_score').float()
            known_scores = stack_root_field(image_roots, 'known_scores').float()
            root_loc = stack_root_field(image_roots, 'localization_quality_probability').float()
            root_integrity = stack_root_field(image_roots, 'instance_integrity_probability').float()
            root_unique = stack_root_field(image_roots, 'uniqueness_probability').float()
            if responsibility_cache is None:
                (native_resp, _, _) = self._soft_gt_responsibility(out['native_candidate_boxes'][b].float().detach(), gt_boxes)
            else:
                native_resp = responsibility_cache[b][0]
            root_resp = native_resp.index_select(0, representatives)
            same_native = same_instance_target[b].detach().float().clamp(0.0, 1.0)
            same_root = same_native.index_select(0, representatives).index_select(1, representatives)
            root_loc_target = localization_target[b].detach().float().index_select(0, representatives)
            root_integrity_target = integrity_target[b].detach().float().index_select(0, representatives)
            matched_root_mask = torch.zeros(len(image_roots), dtype=torch.bool, device=device)
            matched_root_mask[root_idx] = True
            negative_pool = torch.nonzero(~matched_root_mask, as_tuple=False).squeeze(-1)
            matched_labels = gt_labels.index_select(0, gt_idx)
            matched_known = known_scores.index_select(0, root_idx).gather(1, matched_labels[:, None]).squeeze(1).clamp_min(eps)
            matched_unknown = unknown_score.index_select(0, root_idx).clamp_min(eps)
            matched_family_gap = torch.log(matched_known) - torch.log(matched_unknown)
            family_terms.append(F.softplus(family_margin - matched_family_gap))
            family_labels.append(matched_labels)
            family_gaps.append(matched_family_gap.detach())
            for _k in range(int(root_idx.numel())):
                rp = root_idx[_k]
                gi = gt_idx[_k]
                label = matched_labels[_k]
                if negative_pool.numel() == 0:
                    continue
                relatedness = root_resp.index_select(0, negative_pool)[:, gi].clamp(0.0, 1.0)
                danger = unknown_score.detach().index_select(0, negative_pool) * relatedness
                if not bool(relatedness.sum() > eps):
                    continue
                take = min(k, int(negative_pool.numel()))
                top_local = torch.topk(danger, k=take, largest=True).indices
                neg = negative_pool.index_select(0, top_local)
                rel = relatedness.index_select(0, top_local)
                pos_loc_target = root_loc_target[rp]
                pos_integrity_target = root_integrity_target[rp]
                loc_def = (pos_loc_target - root_loc_target.index_select(0, neg)).clamp_min(0.0)
                integrity_def = (pos_integrity_target - root_integrity_target.index_select(0, neg)).clamp_min(0.0)
                duplicate = same_root[rp].index_select(0, neg).clamp(0.0, 1.0)
                w_loc = rel * loc_def
                w_integrity = rel * integrity_def
                w_duplicate = rel * duplicate
                factor_mass = w_loc + w_integrity + w_duplicate
                valid = factor_mass > eps
                if not bool(valid.any()):
                    continue
                neg = neg[valid]
                rel = rel[valid]
                w_loc = w_loc[valid]
                w_integrity = w_integrity[valid]
                w_duplicate = w_duplicate[valid]
                factor_mass = factor_mass[valid]
                beta = float(getattr(getattr(self.cfg, 'decoder', None), 'quality_ranking_exponent', 0.5))
                gamma = float(getattr(getattr(self.cfg, 'decoder', None), 'uniqueness_ranking_exponent', 1.0))
                loc_gap = 0.5 * beta * (torch.log(root_loc[int(rp)].clamp_min(eps)) - torch.log(root_loc.index_select(0, neg).clamp_min(eps)))
                integrity_gap = 0.5 * beta * (torch.log(root_integrity[int(rp)].clamp_min(eps)) - torch.log(root_integrity.index_select(0, neg).clamp_min(eps)))
                duplicate_gap = gamma * (torch.log(root_unique[int(rp)].clamp_min(eps)) - torch.log(root_unique.index_select(0, neg).clamp_min(eps)))
                per_neg = (w_loc * F.softplus(quality_margin - loc_gap) + w_integrity * F.softplus(quality_margin - integrity_gap) + w_duplicate * F.softplus(quality_margin - duplicate_gap)) / factor_mass.clamp_min(eps)
                routed_terms.append((rel * per_neg).sum() / rel.sum().clamp_min(eps))
                routed_labels.append(label)
                routed_gap = (w_loc * loc_gap.detach() + w_integrity * integrity_gap.detach() + w_duplicate * duplicate_gap.detach()) / factor_mass.clamp_min(eps)
                routed_gaps.append((rel * routed_gap).sum() / rel.sum().clamp_min(eps))
                hard_unknown_scores.append((rel * unknown_score.detach().index_select(0, neg)).sum() / rel.sum().clamp_min(eps))
                relatedness_values.append(rel.mean().detach())
                norm = factor_mass.sum().clamp_min(eps)
                loc_weight_values.append((w_loc.sum() / norm).detach())
                integrity_weight_values.append((w_integrity.sum() / norm).detach())
                duplicate_weight_values.append((w_duplicate.sum() / norm).detach())
                pair_count += int(neg.numel())
        family_loss = zero
        if family_terms:
            family_loss = self._macro_class_loss(torch.cat([x.reshape(-1) for x in family_terms]), torch.cat([x.reshape(-1) for x in family_labels]).long())
        quality_loss = zero
        if routed_terms:
            quality_loss = self._macro_class_loss(torch.stack(routed_terms), torch.stack(routed_labels).long())

        def mean_or_zero(values):
            return torch.cat([x.reshape(-1) for x in values]).mean() if values else zero.detach()
        return (family_loss, quality_loss, {'known_unknown_log_gap': mean_or_zero(family_gaps).detach(), 'routed_factor_log_gap': mean_or_zero(routed_gaps).detach(), 'hard_negative_unknown_score': mean_or_zero(hard_unknown_scores).detach(), 'hard_negative_relatedness': mean_or_zero(relatedness_values).detach(), 'localization_route_fraction': mean_or_zero(loc_weight_values).detach(), 'integrity_route_fraction': mean_or_zero(integrity_weight_values).detach(), 'duplicate_route_fraction': mean_or_zero(duplicate_weight_values).detach(), 'pair_count': torch.tensor(pair_count, device=device, dtype=torch.long)})

    def __call__(self, model, out, targets):
        assert 'r11p4_view2_native_boxes' not in out
        assert torch.equal(out['native_candidate_boxes'], out['native_candidate_boxes_before_refinement'])
        roots = self._physical_roots(out)
        matches = self._match_native_candidates(out, targets)
        root_matches = self._match_physical_roots(roots, targets)
        (object_target, _) = self._query_objectness_target_and_class(out, targets)
        (hard, _) = self._r11p4_hardtail_relative_object_ranking_loss(out['query_objectness_logit'], object_target, matches)
        loc = self._native_localization_quality_target(out, targets)
        integ = self._native_instance_integrity_target(out, targets)
        resp = self._build_native_responsibility_cache(out, targets)
        (same, weight) = self._soft_same_instance_target(out, targets, localization_target=loc, responsibility_cache=resp)
        (joint, _) = self._joint_quality_ranking_loss(out, targets, matches, object_target, loc, integ, same_instance_target=same)
        (_, formal, _) = self._formal_continuous_ranking_losses(out, roots, root_matches, targets, loc, integ, same, responsibility_cache=resp)
        values = dict(loss_query_objectness=F.binary_cross_entropy_with_logits(out['query_objectness_logit'].float(), object_target.float()), loss_r11p4_hardtail_relative_object_ranking=hard, loss_localization_quality=self._proper_quality_bce(out['native_candidate_localization_quality_logit'], loc), loss_instance_integrity=self._proper_quality_bce(out['native_candidate_instance_integrity_logit'], integ), loss_soft_same_instance=self._balanced_pair_bce(out['native_candidate_soft_same_instance_evidence_logits'], same, weight), loss_joint_quality_ranking=joint, loss_formal_continuous_related_quality_ranking=formal)
        if model.training:
            model.decoder.native_soft_same_instance.update_empirical_pair_prior(same.detach(), weight.detach())
        losses = {k: float(getattr(self.cfg.loss, k)) * v for (k, v) in values.items()}
        assert set(losses) == set(SEVEN_KEYS)
        return (losses, {})
