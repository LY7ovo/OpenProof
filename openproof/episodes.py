"""First-order class-disjoint QAM readout updates."""
from contextlib import contextmanager
from types import SimpleNamespace
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HEADS = ('native_localization_quality_head', 'native_instance_integrity_head')
QUALITY_KEYS = ('loss_localization_quality', 'loss_instance_integrity', 'loss_joint_quality_ranking', 'loss_formal_continuous_related_quality_ranking')

def readouts(model):
    return [(f'decoder.{head}.2.{name}', p) for head in HEADS for (name, p) in getattr(model.decoder, head)[2].named_parameters()]

def episode(targets, step, num_classes=20):
    labels = [set((int(c) for c in t['labels'].tolist())) for t in targets]
    assert num_classes >= 2
    assert all(labels) and all((s <= set(range(num_classes)) for s in labels))
    options = []
    for classes in sorted({tuple(sorted(s)) for s in labels}):
        held = set(classes)
        support = [i for (i, s) in enumerate(labels) if not s & held]
        query = [i for (i, s) in enumerate(labels) if s <= held]
        if support and query:
            options.append((support, query, classes))
    if not options:
        return None
    (support, query, held) = options[step % len(options)]
    a = set().union(*(labels[i] for i in support))
    b = set().union(*(labels[i] for i in query))
    assert not a & b and (not set(support) & set(query))
    return dict(support=support, query=query, support_classes=sorted(a), heldout_classes=sorted(b), excluded_mixed_images=len(targets) - len(support) - len(query), support_GT=sum((len(targets[i]['labels']) for i in support)), query_GT=sum((len(targets[i]['labels']) for i in query)))

def subset(cached, targets, indices):
    return ({k: v.index_select(0, torch.tensor(indices, device=v.device)) for (k, v) in cached.items()}, [targets[i] for i in indices])

@contextmanager
def preserve_randomness():
    py = random.getstate()
    npstate = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(npstate)

class FastLinear(nn.Module):

    def __init__(self, weight, bias):
        super().__init__()
        self.fast_weight = weight
        self.fast_bias = bias

    def forward(self, x):
        return F.linear(x, self.fast_weight, self.fast_bias)

@contextmanager
def temporary_readouts(model, fast):
    old = []
    try:
        for head in HEADS:
            module = getattr(model.decoder, head)
            old.append((module, module[2]))
            module[2] = FastLinear(fast[f'decoder.{head}.2.weight'], fast[f'decoder.{head}.2.bias'])
        yield
    finally:
        for (module, linear) in old:
            module[2] = linear

def objective(criterion, model, cached, targets):
    model.prepare_batch(targets)
    out = model.perceive_cached(cached)
    (losses, _) = criterion(SimpleNamespace(training=False), out, targets)
    loss = sum((losses[k] for k in QUALITY_KEYS))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite episodic quality objective')
    return loss

def meta_gradients(model, criterion, cached, targets, opt, step, num_classes=20):
    plan = episode(targets, step, num_classes=num_classes)
    params = readouts(model)
    if plan is None:
        return (None, dict(episode_active=0, virtual_image_visits=0))
    lrs = {id(p): g['lr'] for g in opt.param_groups for p in g['params']}
    (support, st) = subset(cached, targets, plan['support'])
    (query, qt) = subset(cached, targets, plan['query'])
    with preserve_randomness():
        try:
            support_loss = objective(criterion, model, support, st)
            grads = torch.autograd.grad(support_loss, [p for (_, p) in params])
            fast = {name: p - lrs[id(p)] * g.detach() for ((name, p), g) in zip(params, grads)}
            with temporary_readouts(model, fast):
                query_loss = objective(criterion, model, query, qt)
                outer = torch.autograd.grad(query_loss, [p for (_, p) in params])
            assert all((bool(torch.isfinite(g).all()) for g in outer))
        finally:
            model.prepare_batch(targets)
    info = dict(plan, episode_active=1, virtual_image_visits=len(st) + len(qt), support_quality_loss=float(support_loss.detach()), query_quality_loss=float(query_loss.detach()), virtual_readout_max_change=max((float((fast[n] - p).detach().abs().max()) for (n, p) in params)))
    return ([g.detach() for g in outer], info)
