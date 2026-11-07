"""Fixed 1035-update OPEN training, Known REM fitting and incremental inheritance."""
from pathlib import Path
import math
import random
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import cache
from .boxes import box_cxcywh_to_xyxy,box_iou
from .config import runtime,read,publish
from .data import Images,source_path,cached_collate
from .detector import FrozenRF
from .model import OpenProof
from .losses import Criterion,SEVEN_KEYS
from .episodes import readouts,meta_gradients
from .reference import ReferenceBank

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
def save(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
def freeze_hidden(model):
    for name in ('native_localization_quality_head','native_instance_integrity_head'):
        getattr(model.decoder,name)[0].requires_grad_(False)
def open_state(model):return {k:v for k,v in model.decoder.state_dict().items() if not k.startswith('reference.')}
def load_open_state(model,state):
    result=model.decoder.load_state_dict(state,strict=False)
    if result.unexpected_keys or any(not key.startswith('reference.') for key in result.missing_keys):
        raise RuntimeError('OPEN checkpoint architecture differs')
def initialize(cfg,task,device,load=False,teacher=False):
    folder=Path(cfg['output'])/task['name'];classes=read(folder/'classes.json');k=len(classes['known_names'])
    if classes['known_names']!=task['known_names']:raise RuntimeError('Checkpoint class semantics differ from the requested task')
    rf=FrozenRF(folder/'rf.pth',k,device,cfg['geometry_selection'])
    model=OpenProof(runtime(k,device),rf)
    if load:
        reference=torch.load(folder/'reference.pth',map_location='cpu',weights_only=True)
        model.decoder.reference.load_state_dict(reference['state'],strict=True)
        row=torch.load(folder/'open_final.pth',map_location='cpu',weights_only=False)
        load_open_state(model,row['decoder'])
    return model,Criterion(model.cfg)

def canonical_sources(query,boxes,logits):
    c,s=boxes[...,:2],boxes[...,2:];lo,hi=c-s*.5,c+s*.5
    inter=(torch.minimum(hi[:,:,None],hi[:,None])-torch.maximum(lo[:,:,None],lo[:,None])).clamp_min(0).prod(-1)
    area=s.prod(-1);overlap=inter/(area[:,:,None]+area[:,None]-inter).clamp_min(torch.finfo(query.dtype).tiny)
    prior=logits.softmax(-1)
    return (overlap*(prior@prior.transpose(1,2))*logits.sigmoid().amax(-1)[:,None,:]).argmax(-1)

def fit_reference(model,features,path):
    if path.is_file():
        model.decoder.reference.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state'],strict=True)
        return
    xs=[];ys=[];ids=[];priors=[];k=model.cfg.num_known_classes;device=model.device_obj
    loader=DataLoader(features,batch_size=32,shuffle=False,num_workers=2,collate_fn=cached_collate)
    with torch.no_grad():
        for cached,targets in loader:
            query=cached['query_summary'].to(device);boxes=cached['box_final'].to(device);logits=cached['native_logits'][...,:k].to(device)
            sources=canonical_sources(query,boxes,logits)
            query,boxes,logits,sources=[v.cpu() for v in (query,boxes,logits,sources)]
            for b,t in enumerate(targets):
                iou,_=box_iou(box_cxcywh_to_xyxy(boxes[b]),box_cxcywh_to_xyxy(t['boxes']))
                eligible=(iou>=.5)&((iou>=.5).sum(-1)[:,None]==1)
                for g,label in enumerate(t['labels']):
                    candidates=eligible[:,g].nonzero().flatten();candidates=candidates[eligible[sources[b,candidates],g]]
                    for source in sources[b,candidates].unique(sorted=True):
                        dest=int(candidates[(sources[b,candidates]==source).nonzero()[0]])
                        xs.append(query[b,source].clone());ys.append(int(label));ids.append(int(t['image_id']));priors.append(logits[b,dest].log_softmax(-1))
    if not xs:raise RuntimeError('No uniquely assigned Known exemplars at IoU .50')
    x=torch.stack(xs);y=torch.tensor(ys);image_ids=torch.tensor(ids);prior=torch.stack(priors).double()
    ref=ReferenceBank(256,k).to(device);ref.fit_references(x.to(device),y.to(device),image_ids.to(device))
    with torch.no_grad():
        banks=1-image_ids.remainder(2)
        distances=torch.cat([ref.raw_normalized_distances(x[s:s+256].to(device),banks[s:s+256].to(device)).cpu() for s in range(0,len(x),256)])
        ref.fit_quantiles(distances.to(device),y.to(device),image_ids.to(device))
        coords=ref.transform_distances(distances.to(device),banks.to(device)).cpu().double()
    ref.cpu().double();ref.requires_grad_(True)
    weights=1/(k*torch.bincount(y,minlength=k).double()[y])
    optimizer=torch.optim.LBFGS(ref.parameters(),max_iter=120,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();logits=ref.logits_from_distances(coords)
        positive=-torch.logsumexp(prior+F.logsigmoid(logits),-1)
        negative=F.softplus(ref.logits_from_distances(coords,~F.one_hot(y,k).bool()).amax(-1))
        loss=(weights*.5*(positive+negative)).sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Known-only REM fit')
        loss.backward();return loss
    optimizer.step(closure);ref.float().requires_grad_(False)
    save(path,dict(state=ref.state_dict(),report=dict(Known_only=True,training_images=len(features),exemplars=len(x),fitted_scalars=3)))
    model.decoder.reference.load_state_dict(ref.state_dict(),strict=True)

def train(cfg,task,previous,device):
    folder=Path(cfg['output'])/task['name'];prepared=Path(cfg['output'])/'prepared'/task['name']
    if (folder/'open_completed.json').is_file():
        if not (folder/'open_final.pth').is_file():raise FileNotFoundError(folder/'open_final.pth')
        return
    seed(cfg['seed']);model,criterion=initialize(cfg,task,device);k=model.cfg.num_known_classes
    if previous:
        old=torch.load(Path(previous)/'open_final.pth',map_location='cpu',weights_only=False)['decoder']
        fresh=model.decoder.state_dict()
        transferable={key:value for key,value in old.items() if not key.startswith('reference.')}
        for key,value in transferable.items():
            if key not in fresh or fresh[key].shape!=value.shape:raise RuntimeError(f'OPEN inheritance changed: {key}')
            fresh[key]=value
        model.decoder.load_state_dict(fresh,strict=True)
        previous_task=next(t for t in cfg['tasks'] if t['name']==Path(previous).name)
        teacher,_=initialize(cfg,previous_task,device,load=True);teacher.eval().requires_grad_(False)
        model._teacher_files=[str(Path(previous)/name) for name in ('rf.pth','open_final.pth','reference.pth')]
        from .incremental import install
        install(model,criterion,teacher)
    dataset=Images(prepared/'open_train.json',source_path(cfg,task['images']))
    scratch=folder/'cache'
    features=cache.build(model,dataset,scratch)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
        fit_reference(model,features,folder/'reference.pth')
    counts=[0]*k
    for index in range(len(dataset)):
        for c in set(dataset.target(index)['labels'].tolist()):counts[c]+=1
    criterion.set_class_balance_counts(counts)
    optimizer=torch.optim.AdamW(model.optimizer_parameter_groups(),weight_decay=1e-4)
    micro_total=max(1035,math.ceil(len(features)/16));generator=torch.Generator().manual_seed(cfg['seed']);batches=[]
    while len(batches)<micro_total:
        order=torch.randperm(len(features),generator=generator).tolist();batches.extend(order[i:i+16] for i in range(0,len(order),16))
    batches=batches[:micro_total];last=folder/'open_last.pth';start=consumed=0
    recipe=dict(classes=k,RF_new_epochs=4,RF_replay_epochs=int(bool(previous)),OPEN_updates=1035,hidden_freeze=312,
        seed=cfg['seed'],routing=True,QAM=True,redundancy=True,REM=True,spatial_odds_train_and_test=True,
        class_disjoint=cfg['class_disjoint'],images=len(features),microbatches=micro_total,loss_weights={key:getattr(model.cfg.loss,key) for key in SEVEN_KEYS})
    publish(folder/'training_config.json',recipe)
    if last.is_file():
        row=torch.load(last,map_location='cpu',weights_only=False)
        if row['recipe']!=recipe:raise RuntimeError('Resume recipe differs')
        load_open_state(model,row['decoder']);optimizer.load_state_dict(row['optimizer'])
        start,consumed=row['updates'],row['consumed']
    if start>=312:freeze_hidden(model)
    loader=DataLoader(features,batch_sampler=batches[consumed:],num_workers=2,collate_fn=cached_collate)
    iterator=iter(loader);model.train();begin=time.perf_counter()
    if last.is_file():
        # Iterator construction seeds workers; restore model RNG after that draw.
        random.setstate(row['python_rng']);np.random.set_state(row['numpy_rng']);torch.set_rng_state(row['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(row['cuda_rng'])
    for step in range(start,1035):
        count=(step+1)*micro_total//1035-step*micro_total//1035
        optimizer.zero_grad(set_to_none=True);params=readouts(model);outer=[torch.zeros_like(p) for _,p in params];active=0;loss_total=0.
        for _ in range(count):
            cached,targets=next(iterator);model.prepare_batch(targets)
            losses,_=criterion(model,model.perceive_cached(cached),targets);total=sum(losses.values())
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite OPEN objective')
            (total/count).backward();loss_total+=float(total.detach())/count
            if cfg['class_disjoint']:
                gradients,_=meta_gradients(model,criterion,cached,targets,optimizer,step,k)
                if gradients is not None:
                    active+=1
                    for dest,gradient in zip(outer,gradients):dest.add_(gradient,alpha=1/count)
        if cfg['class_disjoint']:
            for (_,p),gradient in zip(params,outer):p.grad=gradient if active else None
        torch.nn.utils.clip_grad_norm_([p for p in model.decoder.parameters() if p.requires_grad and p.grad is not None],20.,error_if_nonfinite=True)
        optimizer.step();consumed+=count
        if step+1==312:freeze_hidden(model)
        if (step+1)%10==0 or step==start:
            print(f'[OPEN] {task["name"]} {step+1}/1035 loss={loss_total:.5f}',flush=True)
        if (step+1)%100==0 or step+1 in (1,312,1035):
            save(last,dict(decoder=open_state(model),optimizer=optimizer.state_dict(),updates=step+1,consumed=consumed,recipe=recipe,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []))
            publish(folder/'progress.json',dict(updates=step+1,required_updates=1035,seconds=time.perf_counter()-begin))
    if consumed!=micro_total:raise RuntimeError('Training pool schedule incomplete')
    save(folder/'open_final.pth',dict(decoder=open_state(model),updates=1035,recipe=recipe))
    publish(folder/'open_completed.json',dict(updates=1035,training_images=len(features),full_pool_visited=True))
    del iterator,loader,features
    shutil.rmtree(scratch)
    last.unlink()
    for file in (folder/'rf_training').glob('*/*'):
        if file.is_file() and (file.suffix=='.ckpt' or file.name in ('initial.pth','rf_final.pth')):file.unlink()
