"""Objectness dynamics, fixed physical projection and learned pair relations."""
import torch
from torch import nn
from torch.nn import functional as F

class FrozenRandomProjector(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, seed: int):
        super().__init__()
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))
        weight = torch.randn(output_dim, input_dim, generator=g, dtype=torch.float32)
        weight = F.normalize(weight, dim=1)
        self.register_buffer('weight', weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.detach().float(), self.weight.float())

class NativeQuerySoftSameInstanceHead(nn.Module):

    def __init__(self, dim: int, hidden: int=32, relation_dim: int=24, initial_probability: float=1e-06, prior_momentum: float=0.95):
        super().__init__()
        relation_dim = int(min(max(relation_dim, 8), dim))
        hidden = int(min(max(hidden, 8), 48))
        initial_probability = float(initial_probability)
        prior_momentum = float(prior_momentum)
        if not 0.0 < initial_probability < 0.01:
            raise ValueError('soft same-instance deployment prior must lie in (0,0.01)')
        if not 0.0 <= prior_momentum < 1.0:
            raise ValueError('soft same-instance prior momentum must lie in [0,1)')
        self.prior_momentum = prior_momentum
        self.register_buffer('empirical_pair_prior', torch.tensor(initial_probability, dtype=torch.float32))
        self.project = nn.Linear(dim, relation_dim, bias=False)
        self.net = nn.Sequential(nn.Linear(12, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def update_empirical_pair_prior(self, target: torch.Tensor, weight: torch.Tensor) -> None:
        q = int(target.shape[-1])
        if q <= 1:
            return
        tri = torch.triu(torch.ones((q, q), device=target.device, dtype=torch.bool), diagonal=1)[None]
        valid = weight.float() * tri.float()
        mass = valid.sum()
        if not bool(mass > 1e-08):
            return
        prior = (target.float().clamp(0.0, 1.0) * valid).sum() / mass
        prior = prior.clamp(1e-08, 0.05)
        self.empirical_pair_prior.mul_(self.prior_momentum).add_(prior.detach().to(self.empirical_pair_prior.device) * (1.0 - self.prior_momentum))

    def deployment_logits(self, evidence_logits: torch.Tensor) -> torch.Tensor:
        prior = self.empirical_pair_prior.float().clamp(1e-08, 1.0 - 1e-06)
        prior_logit = torch.log(prior) - torch.log1p(-prior)
        return evidence_logits.float() + prior_logit

    def forward(self, physical_atoms: torch.Tensor, boxes: torch.Tensor, object_logit: torch.Tensor, localization_quality_probability: torch.Tensor, instance_integrity_probability: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.project(physical_atoms.float()), dim=-1, eps=1e-06)
        cosine = torch.matmul(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
        box = boxes.float().detach()
        center = box[..., :2]
        size = box[..., 2:].clamp_min(0.0001)
        dc = center[:, :, None, :] - center[:, None, :, :]
        mean_size = 0.5 * (size[:, :, None, :] + size[:, None, :, :]).clamp_min(0.0001)
        center_sq = (dc / mean_size).square().sum(-1).clamp_max(64.0)
        log_size = size.log()
        scale_l1 = (log_size[:, :, None, :] - log_size[:, None, :, :]).abs().sum(-1).clamp_max(16.0)
        half = 0.5 * size
        lo = center - half
        hi = center + half
        inter_lo = torch.maximum(lo[:, :, None, :], lo[:, None, :, :])
        inter_hi = torch.minimum(hi[:, :, None, :], hi[:, None, :, :])
        inter_wh = (inter_hi - inter_lo).clamp_min(0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]
        area = size[..., 0] * size[..., 1]
        union = area[:, :, None] + area[:, None, :] - inter
        iou = inter / union.clamp_min(1e-08)
        inter_over_min = inter / torch.minimum(area[:, :, None], area[:, None, :]).clamp_min(1e-08)
        containment_i = inter / area[:, :, None].clamp_min(1e-08)
        containment_j = inter / area[:, None, :].clamp_min(1e-08)
        containment_gap = (containment_i - containment_j).abs()
        p = torch.sigmoid(object_logit.float().detach()).clamp(1e-06, 1.0 - 1e-06)
        p_i = p[:, :, None]
        p_j = p[:, None, :]
        q = localization_quality_probability.float().detach().clamp(0.0, 1.0)
        integ = instance_integrity_probability.float().detach().clamp(0.0, 1.0)
        feat = torch.stack((cosine, 1.0 - cosine, center_sq, scale_l1, iou, inter_over_min, containment_gap, torch.minimum(p_i, p_j), p_i * p_j, (p_i - p_j).abs(), torch.minimum(q[:, :, None], q[:, None, :]), torch.minimum(integ[:, :, None], integ[:, None, :])), dim=-1)
        logits = self.net(feat).squeeze(-1)
        return 0.5 * (logits + logits.transpose(1, 2))

class CrossDomainTrajectoryObjectHead(nn.Module):
    INTRINSIC_INDICES = tuple(range(18)) + (21, 22, 23, 24, 25)
    ABSOLUTE_BOX_MOTION_INDICES = (18, 19, 20)

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        if self.feature_dim != 26:
            raise ValueError(f'R11.2 Object head expects the 26-D trajectory contract, got {self.feature_dim}')
        self.intrinsic_dim = len(self.INTRINSIC_INDICES)
        hidden = 64
        self.input_projection = nn.Sequential(nn.Linear(self.feature_dim * 2 + self.intrinsic_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _tie_aware_percentile(x: torch.Tensor) -> torch.Tensor:
        less = (x[:, None, :, :] < x[:, :, None, :]).to(x.dtype).mean(dim=2)
        equal = (x[:, None, :, :] == x[:, :, None, :]).to(x.dtype).mean(dim=2)
        return (2.0 * (less + 0.5 * equal) - 1.0).clamp(-1.0, 1.0)

    @staticmethod
    def _robust_relative(x: torch.Tensor) -> torch.Tensor:
        q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=x.dtype), dim=1)
        (q25, median, q75) = (q[0].unsqueeze(1), q[1].unsqueeze(1), q[2].unsqueeze(1))
        centered = x - median
        iqr = q75 - q25
        rms = centered.square().mean(dim=1, keepdim=True).sqrt()
        scale = torch.where(iqr > 0.0, iqr, torch.where(rms > 0.0, rms, torch.ones_like(rms)))
        relative = centered / scale
        relative = torch.where((iqr > 0.0) | (rms > 0.0), relative, torch.zeros_like(relative))
        return relative.clamp(-8.0, 8.0)

    def relative_coordinates(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.float()
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise RuntimeError(f'R11.2 trajectory Object head expected [B,Q,{self.feature_dim}], got {tuple(x.shape)}')
        return (self._robust_relative(x), self._tie_aware_percentile(x))

    def intrinsic_coordinates(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float()
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise RuntimeError(f'R11.2 trajectory Object head expected [B,Q,{self.feature_dim}], got {tuple(x.shape)}')
        idx = torch.tensor(self.INTRINSIC_INDICES, device=x.device, dtype=torch.long)
        intrinsic = x.index_select(-1, idx)
        if not bool(torch.isfinite(intrinsic).all()):
            raise RuntimeError('R11.2 intrinsic trajectory coordinate contains non-finite values')
        return intrinsic

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (relative, percentile) = self.relative_coordinates(features)
        intrinsic = self.intrinsic_coordinates(features)
        h = self.input_projection(torch.cat((relative, percentile, intrinsic), dim=-1))
        h = self.set_encoder(h)
        return (self.out(h).squeeze(-1), relative)

def trajectory_features(query_layers: torch.Tensor, context_layers: torch.Tensor, box_layers: torch.Tensor) -> torch.Tensor:
    q = query_layers.float()
    c = context_layers.float()
    b = box_layers.float()
    if q.ndim != 4 or c.ndim != 4 or b.ndim != 4:
        raise RuntimeError(f'R11 trajectory inputs must be [L,B,Q,D]/[L,B,Q,4], got {tuple(q.shape)}, {tuple(c.shape)}, {tuple(b.shape)}')
    if q.shape[:3] != c.shape[:3] or q.shape[:3] != b.shape[:3] or b.shape[-1] != 4:
        raise RuntimeError('R11 trajectory inputs are not layer/batch/query aligned')

    def token_stats(x: torch.Tensor) -> torch.Tensor:
        first = x[0]
        final = x[-1]
        endpoint = (F.normalize(first, dim=-1, eps=1e-06) * F.normalize(final, dim=-1, eps=1e-06)).sum(-1)
        if x.shape[0] <= 1:
            z = torch.zeros_like(endpoint)
            o = torch.ones_like(endpoint)
            return torch.stack((o, o, z, z, z, z, endpoint), dim=-1)
        xn = F.normalize(x, dim=-1, eps=1e-06)
        cos = (xn[1:] * xn[:-1]).sum(-1)
        delta = (x[1:] - x[:-1]).norm(dim=-1)
        scale = 0.5 * (x[1:].norm(dim=-1) + x[:-1].norm(dim=-1))
        rel = delta / scale.clamp_min(1e-06)
        return torch.stack((cos.mean(0), cos.amin(0), cos.std(0, unbiased=False), rel.mean(0), rel.amax(0), rel[-1], endpoint), dim=-1)
    qstats = token_stats(q)
    cstats = token_stats(c)
    qn = F.normalize(q, dim=-1, eps=1e-06)
    cn = F.normalize(c, dim=-1, eps=1e-06)
    qc = (qn * cn).sum(-1)
    qcstats = torch.stack((qc.mean(0), qc.amin(0), qc.std(0, unbiased=False), qc[-1]), dim=-1)
    if b.shape[0] <= 1:
        boxstats = torch.zeros((*b.shape[1:3], 8), device=b.device, dtype=b.dtype)
    else:
        step = (b[1:] - b[:-1]).abs()
        step_l1 = step.sum(-1)
        path = step_l1.sum(0)
        endpoint = (b[-1] - b[0]).abs().sum(-1)
        last = step_l1[-1]
        convergence = last / path.clamp_min(1e-06)
        straightness = torch.where(path > 0.0, (endpoint / path).clamp(0.0, 1.0), torch.ones_like(path))
        tortuosity = (-torch.log(straightness.clamp_min(1e-06))).clamp_max(8.0)
        center_path = step[..., :2].sum(-1).sum(0)
        center_fraction = center_path / path.clamp_min(1e-06)
        area = (b[..., 2].clamp_min(1e-05) * b[..., 3].clamp_min(1e-05)).log()
        area_change = (area[-1] - area[0]).abs()
        aspect = (b[..., 2].clamp_min(1e-05) / b[..., 3].clamp_min(1e-05)).log()
        aspect_change = (aspect[-1] - aspect[0]).abs()
        boxstats = torch.stack((path, endpoint, last, convergence, tortuosity, center_fraction, area_change, aspect_change), dim=-1)
    result = torch.cat((qstats, cstats, qcstats, boxstats), dim=-1)
    if result.shape[-1] != 26 or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f'R11 trajectory feature contract failed: shape={tuple(result.shape)} finite={bool(torch.isfinite(result).all())}')
    return result
