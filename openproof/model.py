"""OpenProof: objectness, QAM, full IRM and full REM on frozen RF queries."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .heads import FrozenRandomProjector, CrossDomainTrajectoryObjectHead, NativeQuerySoftSameInstanceHead, trajectory_features
from .relations import routed_identity, relation_odds_probability
from .reference import ReferenceBank


class RootList(list):
    def __init__(self,fields):self.fields=fields
    def __len__(self):return len(self.fields['box'])
    def __bool__(self):return bool(len(self))
    def __getitem__(self,index):return {name:value[index] for name,value in self.fields.items()}
    def __iter__(self):return (self[i] for i in range(len(self)))


def stack_root_field(roots,field):return roots.fields[field]


def make_roots(out):
    """Each frozen candidate is one physical root; no hard merge or NMS."""
    boxes=out['native_candidate_boxes'];roots=[]
    names={'ranking_object_probability':'ranking_object_probability','known_scores':'known_scores',
           'unknown_ranking_score':'unknown_score','localization_quality_probability':'localization_quality_probability',
           'instance_integrity_probability':'instance_integrity_probability','uniqueness_probability':'uniqueness_probability'}
    for b in range(len(boxes)):
        fields=dict(box=boxes[b],representative=torch.arange(boxes.shape[1],device=boxes.device))
        fields.update({name:out['native_candidate_'+key][b] for name,key in names.items()})
        roots.append(RootList(fields))
    return roots


class Decoder(nn.Module):
    def __init__(self,classes):
        super().__init__();self.num_classes=classes;self.num_queries=300
        self.quality_ranking_exponent=.5;self.uniqueness_ranking_exponent=1.;self.uniqueness_temperature=1.
        self.physical_atomizer=FrozenRandomProjector(524,256,14942)
        self.r11_query_objectness_head=CrossDomainTrajectoryObjectHead(26)
        self.native_localization_quality_head=nn.Sequential(nn.Linear(260,128),nn.GELU(),nn.Linear(128,1))
        self.native_instance_integrity_head=nn.Sequential(nn.Linear(261,128),nn.GELU(),nn.Linear(128,1))
        for head in (self.native_localization_quality_head,self.native_instance_integrity_head):
            nn.init.zeros_(head[-1].weight);nn.init.constant_(head[-1].bias,float(torch.logit(torch.tensor(.95))))
        self.native_soft_same_instance=NativeQuerySoftSameInstanceHead(256,32,24,1e-6,.95)
        self.reference=ReferenceBank(256,classes)
        self.reference.requires_grad_(False);self.reference_banks=None

    def forward(self,cached):
        q=cached['query_summary'];c=cached['context_summary'];first=cached['box_first'];boxes=cached['box_final']
        raw=torch.cat((q,c,first,boxes,boxes-first),-1)
        physical=self.physical_atomizer(raw)
        obj,_=self.r11_query_objectness_head(cached['object_trajectory_features'])
        loc_logit=self.native_localization_quality_head(torch.cat((physical,boxes),-1)).squeeze(-1)
        loc=loc_logit.sigmoid()
        int_logit=self.native_instance_integrity_head(torch.cat((physical,boxes,loc.detach()[...,None]),-1)).squeeze(-1)
        integrity=int_logit.sigmoid()
        pair=self.native_soft_same_instance(physical,boxes,obj,loc,integrity)
        deployed=self.native_soft_same_instance.deployment_logits(pair)
        rank=obj+torch.logit(loc.clamp(1e-6,1-1e-6))+torch.logit(integrity.clamp(1e-6,1-1e-6))
        unique,redundancy,pair_probability=relation_odds_probability(deployed,rank,boxes,1.)
        logits=cached['native_logits'][...,:self.num_classes]
        observation,source=routed_identity(q,boxes,boxes,logits,pair)
        banks=self.reference_banks[:,None] if self.training and self.reference_banks is not None else None
        with torch.no_grad():validity=self.reference.probabilities(observation,banks)
        prior=logits.detach().log_softmax(-1)
        accepted=(prior.softmax(-1)*validity).sum(-1).clamp(1e-6,1-1e-6)
        presence=obj.sigmoid();support=presence*(loc*integrity).clamp_min(1e-12).pow(.25)*unique
        known=support[...,None]*accepted[...,None]*prior.softmax(-1)
        unknown=support*(1-accepted)
        reference=(-torch.logit(accepted)-__import__('math').log(self.num_classes))[...,None].expand_as(prior)
        out=dict(query_objectness_logit=obj,native_candidate_boxes=boxes,
            native_candidate_boxes_before_refinement=boxes,native_candidate_object_logit=obj,
            native_candidate_localization_quality_logit=loc_logit,native_candidate_localization_quality_probability=loc,
            native_candidate_instance_integrity_logit=int_logit,native_candidate_instance_integrity_probability=integrity,
            native_candidate_soft_same_instance_evidence_logits=pair,native_candidate_soft_same_instance_logits=deployed,
            native_candidate_soft_same_instance_probability=pair_probability,native_candidate_uniqueness_probability=unique,
            native_candidate_redundancy_evidence=redundancy,native_candidate_known_leaf_logits=prior,
            native_candidate_known_reference_logits=reference,native_candidate_novelty_log_evidence=torch.zeros_like(obj),
            native_candidate_existence_probability=presence,native_candidate_ranking_object_probability=support,
            native_candidate_known_scores=known,native_candidate_unknown_score=unknown,
            native_candidate_unknown_ranking_score=unknown,anchor13_observation=observation,anchor13_validity=validity,
            instance_route_source=source,scores=torch.cat((known,unknown[...,None]),-1))
        return out


class OpenProof(nn.Module):
    def __init__(self,cfg,rf=None):
        super().__init__();self.cfg=cfg;self.device_obj=torch.device(cfg.train.device)
        self.rf=rf;self.decoder=Decoder(cfg.num_known_classes).to(self.device_obj)
    def train(self,mode=True):
        super().train(mode)
        if self.rf is not None:self.rf.eval()
        return self
    def prepare_batch(self,targets):
        self.decoder.reference_banks=torch.tensor([1-int(t['image_id'])%2 for t in targets],device=self.device_obj)
    @torch.no_grad()
    def extract_frozen_native_cache(self,images):
        n=self.rf.native_pass(images,capture_cross_attention=True)
        summary=lambda x:x[-1] if len(x)==1 else x[-1]+(x[1:]-x[:-1]).mean(0)
        return dict(query_summary=summary(n['decoder_hidden_layers']).detach(),
            context_summary=summary(n['decoder_cross_context_layers']).detach(),
            box_first=n['decoder_box_layers'][0].detach(),box_final=n['decoder_box_layers'][-1].detach(),
            object_trajectory_features=trajectory_features(n['decoder_hidden_layers'],n['decoder_cross_context_layers'],n['decoder_box_layers']).detach(),
            native_logits=n['native_logits'].detach())
    def perceive_cached(self,cached):
        return self.decoder({k:v.to(self.device_obj,non_blocking=True) for k,v in cached.items()})
    def cached_support_atoms(self,cached):
        a={k:v.to(self.device_obj) for k,v in cached.items()}
        raw=torch.cat((a['query_summary'],a['context_summary'],a['box_first'],a['box_final'],a['box_final']-a['box_first']),-1)
        return self.decoder.physical_atomizer(raw),a['box_final']
    def optimizer_parameter_groups(self):
        relation=list(self.decoder.native_soft_same_instance.parameters());used={id(p) for p in relation}
        common=[p for p in self.decoder.parameters() if p.requires_grad and id(p) not in used]
        return [dict(params=common,lr=1e-4,name='geometry_quality'),dict(params=relation,lr=1e-3,name='soft_same')]
