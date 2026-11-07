"""Joint candidate-by-class TopK prediction and benchmark-specific AP protocols."""
from collections import defaultdict
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import read,publish
from .data import Images,source_path,image_collate
from .boxes import box_cxcywh_to_xyxy
from .training import initialize

def select(boxes,scores,image_id,top_k):
    values=scores.detach().cpu().numpy().reshape(-1);classes=scores.shape[-1]
    order=np.arange(len(values));labels=order%classes
    chosen=np.lexsort((order,labels,-values))[:top_k]
    b=boxes.detach().cpu().tolist()
    return [dict(image_id=int(image_id),box=b[int(i)//classes],score=float(values[i]),label=int(labels[i])) for i in chosen]

def predict(cfg,task,test,device):
    folder=Path(cfg['output'])/task['name'];out=folder/'tests'/test['name'];out.mkdir(parents=True,exist_ok=True)
    dataset=Images(Path(cfg['output'])/'prepared'/task['name']/('test_'+test['name']+'.json'),source_path(cfg,test.get('images',task['images'])))
    if len(dataset)!=test['expected_images']:raise ValueError('Full benchmark image count differs')
    model,_=initialize(cfg,task,device,load=True);model.eval()
    identity=dict(recipe=read(folder/'training_config.json'),checkpoint_mtime=(folder/'open_final.pth').stat().st_mtime_ns,
        test_annotations=dataset.payload,top_k=test['top_k'],geometry=cfg['geometry_selection'])
    # Keep the full annotation payload in its own dataset file, not in each shard.
    import hashlib
    identity['test_annotations']=hashlib.sha256(dataset.annotation.read_bytes()).hexdigest()
    identity['reference']=hashlib.sha256((folder/'reference.pth').read_bytes()).hexdigest()
    if (out/'manifest.json').exists():
        old=read(out/'manifest.json')
        if old['identity']!=identity:raise RuntimeError('Existing predictions belong to a different model or protocol')
        if old['complete']:return out
    publish(out/'manifest.json',dict(identity=identity,complete=False,images=0))
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=2,collate_fn=image_collate)
    seen=[]
    with torch.inference_mode():
        for index,(images,targets) in enumerate(loader):
            file=out/'shards'/f'{index:05d}.json';ids=[int(t['image_id']) for t in targets]
            if file.exists():
                row=read(file)
                if row['ids']!=ids:raise RuntimeError('Resume shard image IDs differ')
            else:
                cached=model.extract_frozen_native_cache(images);values=model.perceive_cached(cached);pred=[]
                for boxes,scores,t in zip(values['native_candidate_boxes'],values['scores'],targets):
                    h,w=t['orig_size'].tolist();scale=boxes.new_tensor([w,h,w,h])
                    xy=(box_cxcywh_to_xyxy(boxes)*scale).clamp_min(0).clamp(max=scale)
                    pred.extend(select(xy,scores,int(t['image_id']),test['top_k']))
                row=dict(ids=ids,predictions=pred);publish(file,row)
            seen.extend(ids)
            if index%25==0:print(f'[test] {task["name"]}/{test["name"]} {len(seen)}/{len(dataset)}',flush=True)
    if len(seen)!=len(set(seen)) or set(seen)!={int(im['id']) for im in dataset.images}:raise RuntimeError('Test coverage incomplete')
    publish(out/'manifest.json',dict(identity=identity,complete=True,images=len(seen),predictions=len(seen)*test['top_k']))
    return out

def coco_ap(data,predictions,known,top_k):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    data=dict(data,info={},licenses=[])
    submission=[dict(image_id=p['image_id'],category_id=p['label']+1,score=p['score'],
                     bbox=[p['box'][0],p['box'][1],p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]]) for p in predictions]
    with redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=data;gt.createIndex()
        if submission:dt=gt.loadRes(submission)
        else:
            dt=COCO();dt.dataset=dict(data,annotations=[]);dt.createIndex()
        evaluator=COCOeval(gt,dt,'bbox');evaluator.params.recThrs=np.linspace(0,1,101)
        evaluator.params.maxDets=[1,10,100];evaluator.evaluate();evaluator.accumulate()
    precision=evaluator.eval['precision'][:,:,:,0,-1]
    ids=list(evaluator.params.catIds)
    def mean(values):
        usable=values[values>=0];return float(usable.mean()) if len(usable) else None
    indices=[i for i,c in enumerate(ids) if c<=known];unknown=ids.index(known+1)
    rec=evaluator.eval['recall'][0,unknown,0,-1]
    return dict(K_AP=mean(precision[:,:,indices]),K_AP50=mean(precision[0,:,indices]),
        U_AP=mean(precision[:,:,unknown]),U_AP50=mean(precision[0,:,unknown]),U_Recall=float(rec) if rec>=0 else None)

def unknown_author(data,predictions,k,expected=None):
    from .evaluation.unknown_voc import voc_evaluate
    images={int(a['image_id']) for a in data['annotations'] if int(a['category_id'])==k+1}
    if expected is not None and len(images)!=expected:raise RuntimeError('Author Unknown image scope differs from the benchmark')
    if not images:return dict(U_AP50=None,U_Recall=None,U_Precision=None,U_F1=None,unknown_images=0)
    anns={str(i):[] for i in images};dets={k:defaultdict(list)}
    for a in data['annotations']:
        if int(a['image_id']) not in images:continue
        x,y,w,h=a['bbox'];anns[str(a['image_id'])].append([x,y,x+w,y+h,int(a['category_id'])-1])
    for p in predictions:
        if p['image_id'] in images and p['label']==k:dets[k][str(p['image_id'])].append([*p['box'],p['score']])
    for image_id in images:dets[k][str(image_id)]=np.asarray(dets[k][str(image_id)],dtype=float).reshape(-1,5)
    result=voc_evaluate(dets,anns,k,ovthresh=.5,use_07_metric=True)
    recall,precision,ap=result[:3]
    return dict(U_AP50=float(ap),U_Recall=float(recall),U_Precision=float(precision),
        U_F1=float(2*recall*precision/(recall+precision)) if recall+precision else 0.,unknown_images=len(images))

def owod_ap(cfg,task,test,data,predictions,out):
    from .evaluation.owod_voc import voc_eval
    names=task['known_names'];k=len(names);old=read(Path(cfg['output'])/'prepared'/task['name']/'classes.json')['previous_known']
    ann=out/'voc/Annotations';ann.mkdir(parents=True,exist_ok=True);det=out/'voc/detections';det.mkdir(exist_ok=True)
    xml=source_path(cfg,test['voc_xml'])
    for im in data['images']:
        oid=im.get('owod_id',im.get('original_id',Path(im['file_name']).stem));source=xml/(str(oid)+'.xml')
        if not source.is_file():raise FileNotFoundError(f'Original complete OWOD XML is required: {source}')
        shutil.copyfile(source,ann/(str(im['id'])+'.xml'))
    ids=out/'voc/test.txt';ids.write_text(''.join(str(im['id'])+'\n' for im in data['images']))
    labels=names+(['unknown'] if test['unknown'] else []);lines=defaultdict(list)
    for p in predictions:
        if p['label']>=len(labels):continue
        x,y,x2,y2=p['box'];lines[p['label']].append(f'{p["image_id"]} {p["score"]:.3f} {x+1:.1f} {y+1:.1f} {x2:.1f} {y2:.1f}\n')
    for c,name in enumerate(labels):(det/(name+'.txt')).write_text(''.join(lines[c]))
    rows=[];ose=0;closed=[];opened=[];nearest=[]
    for c,name in enumerate(labels):
        rec,prec,ap,count,nunk,cs,op=voc_eval(str(det/'{}.txt'),str(ann/'{}.xml'),str(ids),name,ovthresh=.5,use_07_metric=False,known_classes=tuple(names))
        rows.append(dict(name=name,AP=float(ap),recall=float(rec[-1]) if len(rec) else 0.))
        if c<k:
            ose+=int(count)
            if len(rec):
                pos=int(np.argmin(np.abs(rec-.8)));closed.append(float(cs[pos]));opened.append(float(op[pos]));nearest.append(float(rec[pos]))
    mean=lambda values:float(np.mean(values)) if len(values) else None
    result=dict(K_AP50=mean([r['AP'] for r in rows[:k]]),previous_K_AP50=mean([r['AP'] for r in rows[:old]]),
        new_K_AP50=mean([r['AP'] for r in rows[old:k]]),U_AP50=rows[-1]['AP'] if test['unknown'] else None,
        U_Recall=rows[-1]['recall'] if test['unknown'] else None,A_OSE=ose,
        WI_08=sum(opened)/sum(closed) if sum(closed)>0 and test['unknown'] else None,
        WI_Known_recall_nearest=nearest,per_class=rows)
    known,recall=result['K_AP50'],result['U_Recall']
    result['H']=2*known*recall/(known+recall) if recall is not None and known+recall else None
    return result

def metric_protocol(test):
    if test['protocol']=='owod':
        known=dict(interpolation='VOC2010 continuous',IoU=.5,matching='author inclusive-pixel IoU > .50; difficult handling retained')
        return dict(Known=known,Unknown=dict(known) if test['unknown'] else None)
    known=dict(interpolation='COCO101',recall_points=101,IoU=[.5,.95],AP50_IoU=.5,
               matching='COCO crowd/ignore; maxDets100 per image/category')
    if not test['unknown']:unknown=None
    elif test['protocol']=='uod':
        unknown=dict(interpolation='VOC07 AP11',recall_points=11,IoU=.5,
                     matching='author inclusive-pixel IoU > .50',image_scope='images containing Unknown GT')
    else:unknown=dict(known)
    return dict(Known=known,Unknown=unknown)


def score(cfg,task,test):
    out=Path(cfg['output'])/task['name']/'tests'/test['name'];manifest=read(out/'manifest.json')
    if not manifest['complete']:raise RuntimeError('Complete predictions required')
    data=read(Path(cfg['output'])/'prepared'/task['name']/('test_'+test['name']+'.json'))
    pred=[];ids=[]
    for file in sorted((out/'shards').glob('*.json')):
        row=read(file);ids.extend(row['ids']);pred.extend(row['predictions'])
    if len(ids)!=len(set(ids)) or len(ids)!=manifest['images'] or len(pred)!=manifest['predictions']:raise RuntimeError('Prediction shards incomplete')
    k=len(task['known_names'])
    if test['protocol']=='owod':metrics=owod_ap(cfg,task,test,data,pred,out)
    else:
        metrics=coco_ap(data,pred,k,test['top_k'])
        if test['protocol']=='uod':
            # UOD reports Unknown AP50 from its author evaluator, not COCO U-AP.
            metrics['U_AP']=None
            metrics.update(unknown_author(data,pred,k,test.get('expected_unknown_images')))
    if not test['unknown']:
        for key in ('U_AP','U_AP50','U_Recall','U_Precision','U_F1','H'):metrics[key]=None
    result=dict(benchmark=cfg['benchmark'],task=task['name'],test=test['name'],metrics=metrics,
        AP_protocol=metric_protocol(test),test_images=len(ids),output_budget=test['top_k'],protocol=test['protocol'])
    target=out/'metrics.json'
    if target.is_file():
        previous=read(target)
        if previous.get('AP_protocol')!=result['AP_protocol']:
            backup=out/'metrics_before_protocol_fix.json'
            if not backup.exists():shutil.copyfile(target,backup)
    publish(target,result)
    print('[result]',json_format(result),flush=True)
    return result

def json_format(value):
    import json
    return json.dumps(value,ensure_ascii=False)
