"""Full Known exemplar memory, cross-image distances and three-scalar REM."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceBank(nn.Module):
    def __init__(self,dim,classes,min_scale=.02):
        super().__init__()
        if classes<3:raise ValueError('Withdrawal fitting requires at least three Known categories')
        self.dim,self.classes,self.min_scale=dim,classes,min_scale
        initial=math.log(math.expm1(1.))
        self.gap_raw=nn.Parameter(torch.tensor(initial));self.distance_raw=nn.Parameter(torch.tensor(initial));self.bias=nn.Parameter(torch.tensor(1.))
        self.register_buffer('observation_mean',torch.zeros(dim));self.register_buffer('observation_scale',torch.ones(dim))
        self.register_buffer('exemplar_observations',torch.empty(0,dim));self.register_buffer('exemplar_labels',torch.empty(0,dtype=torch.long))
        self.register_buffer('exemplar_image_banks',torch.empty(0,dtype=torch.long));self.register_buffer('positive_distance_scale',torch.ones(2,classes))
        self.register_buffer('quantile_knots',torch.empty(2,classes,0));self.register_buffer('quantile_coordinates',torch.empty(2,classes,0))
        self.register_buffer('quantile_counts',torch.zeros(2,classes,dtype=torch.long))
    def whiten(self,x):return (x.float()-self.observation_mean)/self.observation_scale
    @torch.no_grad()
    def raw_distances(self,z,banks):
        q=self.whiten(z.detach());r=self.whiten(self.exemplar_observations)
        result=q.new_full((len(q),self.classes),torch.inf)
        for start in range(0,len(r),512):
            stop=start+512
            distance=torch.cdist(q,r[start:stop]).square()/self.dim
            distance.masked_fill_(banks[:,None]!=self.exemplar_image_banks[None,start:stop],torch.inf)
            result.scatter_reduce_(1,self.exemplar_labels[None,start:stop].expand(len(q),-1),distance,reduce='amin',include_self=True)
        if not torch.isfinite(result).all():raise RuntimeError('Missing Known exemplars in a class/fold')
        return result
    @torch.no_grad()
    def raw_normalized_distances(self,observations,banks):
        shape=observations.shape[:-1];z=observations.reshape(-1,self.dim)
        bank=banks.expand(shape).reshape(-1).to(z.device)
        rows=[self.raw_distances(z[s:s+128],bank[s:s+128]) for s in range(0,len(z),128)]
        ratio=torch.cat(rows)/self.positive_distance_scale[bank]
        return ratio.clamp_min(1e-6).log().reshape(*shape,self.classes)
    @torch.no_grad()
    def fit_references(self,observations,labels,image_ids):
        device=self.observation_mean.device;x=observations.float().to(device);y=labels.to(device);ids=image_ids.to(device)
        self.observation_mean.copy_(x.mean(0));self.observation_scale.copy_(x.var(0,unbiased=False).sqrt().clamp_min(self.min_scale))
        self.exemplar_observations=x.detach().clone();self.exemplar_labels=y.detach().clone();self.exemplar_image_banks=ids.remainder(2)
        for b in (0,1):
            counts=torch.bincount(y[ids.remainder(2)==b],minlength=self.classes)
            if not bool((counts>=2).all()):raise RuntimeError(f'Insufficient Known exemplar support in fold {b}: {counts.tolist()}')
        banks=1-ids.remainder(2)
        raw=torch.cat([self.raw_distances(x[s:s+128],banks[s:s+128]) for s in range(0,len(x),128)])
        own=raw.gather(1,y[:,None]).squeeze(1)
        for b in (0,1):
            for c in range(self.classes):self.positive_distance_scale[b,c]=own[(banks==b)&(y==c)].mean().clamp_min(1e-6)
    def transform_distances(self,raw,banks):
        shape=raw.shape;flat=raw.reshape(-1,self.classes);bank=banks.expand(shape[:-1]).reshape(-1).to(raw.device)
        result=torch.empty_like(flat)
        for b in (0,1):
            selected=bank==b
            if not bool(selected.any()):continue
            x=flat[selected]
            for c in range(self.classes):
                n=int(self.quantile_counts[b,c])
                if not n:raise RuntimeError('Fit Known distance quantiles first')
                knots=self.quantile_knots[b,c,:n];coords=self.quantile_coordinates[b,c,:n];value=x[:,c].contiguous()
                if n==1:y=coords[0]+value-knots[0]
                else:
                    hi=torch.searchsorted(knots.contiguous(),value).clamp(1,n-1);lo=hi-1
                    y=coords[lo]+(value-knots[lo])/(knots[hi]-knots[lo])*(coords[hi]-coords[lo])
                    y=torch.where(value<knots[0],coords[0]+value-knots[0],y)
                    y=torch.where(value>knots[-1],coords[-1]+value-knots[-1],y)
                result[selected,c]=y
        return torch.stack((raw,result.reshape(shape)),-1)
    @torch.no_grad()
    def fit_quantiles(self,distances,labels,image_ids):
        banks=1-image_ids.remainder(2);tables={};maximum=0
        for b in (0,1):
            for c in range(self.classes):
                values=distances[(banks==b)&(labels==c),c].sort().values
                if not len(values):raise RuntimeError(f'Missing calibration class/fold {c}/{b}')
                knots,counts=values.unique_consecutive(return_counts=True)
                coords=torch.logit((counts.cumsum(0)-counts*.5)/len(values));tables[b,c]=(knots,coords);maximum=max(maximum,len(knots))
        self.quantile_knots=distances.new_zeros(2,self.classes,maximum);self.quantile_coordinates=torch.zeros_like(self.quantile_knots)
        for (b,c),(knots,coords) in tables.items():
            n=len(knots);self.quantile_counts[b,c]=n;self.quantile_knots[b,c,:n]=knots;self.quantile_coordinates[b,c,:n]=coords
    def logits_from_distances(self,distances,available=None):
        raw,absolute=distances[...,0],distances[...,1]
        allowed=torch.ones_like(raw,dtype=torch.bool) if available is None else available.to(raw.device).bool().expand_as(raw)
        nearest,indices=raw.masked_fill(~allowed,torch.inf).topk(2,dim=-1,largest=False,sorted=True)
        if not torch.isfinite(nearest[...,1]).all():raise RuntimeError('At least two available reference categories required')
        rival=torch.where(torch.arange(self.classes,device=raw.device)==indices[...,:1],nearest[...,1:2],nearest[...,:1])
        return (self.bias+F.softplus(self.gap_raw)*(rival-raw)-F.softplus(self.distance_raw)*absolute).masked_fill(~allowed,-torch.inf)
    def probabilities(self,observations,banks=None):
        if banks is not None:return self.logits_from_distances(self.transform_distances(self.raw_normalized_distances(observations,banks),banks)).sigmoid()
        probs=[]
        for b in (0,1):
            banks=torch.full(observations.shape[:-1],b,device=observations.device,dtype=torch.long)
            probs.append(self.probabilities(observations,banks))
        return (.5*(probs[0]+probs[1])).clamp(1e-6,1-1e-6)
    def _load_from_state_dict(self,state,prefix,metadata,strict,missing,unexpected,errors):
        for name in ('exemplar_observations','exemplar_labels','exemplar_image_banks','quantile_knots','quantile_coordinates'):
            if prefix+name in state:setattr(self,name,torch.empty_like(state[prefix+name],device=self.observation_mean.device))
        super()._load_from_state_dict(state,prefix,metadata,strict,missing,unexpected,errors)
