"""One RF policy: new-class four epochs, then official replay one epoch."""
from contextlib import contextmanager
from pathlib import Path
import torch
from .config import read,publish
from .data import detector_dataset,source_path
from .inheritance import expand_detector_state

def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
def state(payload):
    value=payload['args'];return dict(value) if isinstance(value,dict) else vars(value)
def mask_old_logits(output,old):
    if isinstance(output,dict):return {k:(torch.cat((torch.full_like(v[...,:old],-20.),v[...,old:]),-1) if k=='pred_logits' else mask_old_logits(v,old)) for k,v in output.items()}
    if isinstance(output,list):return [mask_old_logits(v,old) for v in output]
    if isinstance(output,tuple):return tuple(mask_old_logits(v,old) for v in output)
    return output
@contextmanager
def old_label_mask(old):
    from rfdetr.training import RFDETRModelModule
    original=RFDETRModelModule.__init__
    def init(instance,*args,**kwargs):
        original(instance,*args,**kwargs)
        instance.model.register_forward_hook(lambda module,inputs,out:mask_old_logits(out,old) if module.training and old else out)
    RFDETRModelModule.__init__=init
    try:yield
    finally:RFDETRModelModule.__init__=original

def progress(folder,epochs):
    files=list(Path(folder).glob('*.ckpt'))
    if not files:return None,None
    checkpoint=max(files,key=lambda p:p.stat().st_mtime_ns)
    value=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    fit=value.get('loops',{}).get('fit_loop',{});total=fit.get('epoch_progress',{}).get('total',{})
    complete=int(value.get('epoch',-1))>=epochs-1 and int(total.get('processed',0))>=epochs
    for name in ('epoch_loop.batch_progress','epoch_loop.val_loop.batch_progress'):
        p=fit.get(name,{});c=p.get('current',{})
        complete&=p.get('is_last_batch') is True and c.get('ready',0)>0 and c.get('ready')==c.get('completed')
    return checkpoint,value if complete else None

def train(cfg,task,previous,device):
    from rfdetr import RFDETRSmall
    folder=Path(cfg['output'])/task['name'];prepared=Path(cfg['output'])/'prepared'/task['name']
    classes=read(prepared/'classes.json');names=classes['known_names'];old=classes['previous_known']
    anchor=folder/'rf.pth'
    if (folder/'rf_completed.json').is_file():
        if not anchor.is_file():raise FileNotFoundError(anchor)
        return anchor
    source=Path(previous)/'rf.pth' if old else Path(cfg['pretrained'])
    if not source.is_file():raise FileNotFoundError(f'Pretrained detector missing: {source}')
    if old and not (Path(previous)/'open_completed.json').is_file():raise RuntimeError('Previous full task must finish first')
    for phase,epochs in [('new',4)]+([('replay',1)] if old else []):
        out=folder/'rf_training'/phase;out.mkdir(parents=True,exist_ok=True)
        final=out/'rf_final.pth'
        if (out/'completed.json').is_file():
            if not final.is_file():raise FileNotFoundError(final)
            source=final;continue
        seed=out/'initial.pth'
        if not seed.is_file():
            torch.manual_seed(cfg['seed']+int(task.get('task',1)))
            api=RFDETRSmall(pretrain_weights=str(source),num_classes=len(names),device='cpu',resolution=512)
            core=api.model.model;incoming=torch.load(source,map_location='cpu',weights_only=False)
            if phase=='new':
                for name,module in core.named_modules():
                    if (name=='class_embed' or name.startswith('transformer.enc_out_class_embed.')) and isinstance(module,torch.nn.Linear):
                        torch.nn.init.normal_(module.weight,std=.01);torch.nn.init.constant_(module.bias,-4.59511985013459)
                if old:
                    expanded,_=expand_detector_state(incoming['model'],core.state_dict(),read(Path(previous)/'classes.json')['known_names'],names)
                    core.load_state_dict(expanded,strict=True)
            else:core.load_state_dict(incoming['model'],strict=True)
            args=vars(api.model.args);args.update(class_names=names,num_classes=len(names))
            save(seed,dict(model=core.state_dict(),args=args,model_config=api.model_config.model_dump(),model_name='rf-detr-small',rfdetr_version='1.9.3'))
        resume,done=progress(out,epochs)
        if done is None:
            data=detector_dataset(prepared/('train_new.json' if phase=='new' else 'replay.json'),
                source_path(cfg,task['images']),out/'data')
            api=RFDETRSmall(pretrain_weights=str(seed),num_classes=len(names),device='cpu',resolution=512)
            options=dict(dataset_dir=str(data),dataset_file='roboflow',output_dir=str(out),epochs=epochs,
                batch_size=8,grad_accum_steps=2,num_workers=2,pin_memory=True,persistent_workers=True,
                prefetch_factor=2,lr=1e-4,lr_encoder=1.5e-4,lr_drop=max(epochs,2),seed=cfg['seed'],
                multi_scale=True,expanded_scales=True,use_ema=False,early_stopping=False,run_test=False,
                eval_interval=1,checkpoint_interval=1,tensorboard=False,wandb=False,progress_bar='tqdm',class_names=names,device=device)
            if resume:options['resume']=str(resume)
            publish(out/'options.json',options)
            with old_label_mask(old if phase=='new' else 0):api.train(**options)
            _,done=progress(out,epochs)
            if done is None:raise RuntimeError('No native checkpoint proves the fixed RF budget completed')
        payload=torch.load(seed,map_location='cpu',weights_only=False)
        weights={k[len('model.'):]:v for k,v in done['state_dict'].items() if k.startswith('model.')}
        if weights.keys()!=payload['model'].keys():raise RuntimeError('Native RF checkpoint architecture differs')
        payload['model']=weights;save(final,payload)
        publish(out/'completed.json',dict(phase=phase,epochs=epochs,known_names=names,old_logits_masked=bool(old and phase=='new'),
            optimizer_updates=int(done.get('global_step',0)),selection='fixed-budget last checkpoint'))
        source=final
    payload=torch.load(source,map_location='cpu',weights_only=False);save(anchor,payload)
    publish(folder/'classes.json',classes)
    publish(folder/'rf_completed.json',dict(new_epochs=4,replay_epochs=int(bool(old)),known_names=names))
    return anchor
