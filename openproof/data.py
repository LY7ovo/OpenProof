"""COCO annotation preparation and deterministic frozen-detector image loading."""
from collections import defaultdict
from pathlib import Path
import copy
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from .boxes import box_xyxy_to_cxcywh
from .config import read,publish

def source_path(cfg,path):
    p=Path(path);return p if p.is_absolute() else Path(cfg['data_root'])/p

def convert(source,names,training=False,new_names=None):
    aliases={'airplane':'aeroplane','dining table':'diningtable','motorcycle':'motorbike','potted plant':'pottedplant','couch':'sofa','tv':'tvmonitor'}
    payload=read(source);lookup={int(c['id']):aliases.get(c['name'],c['name']) for c in payload['categories']}
    mapping={name:i+1 for i,name in enumerate(names)}
    images=copy.deepcopy(payload['images']);ids=[int(im['id']) for im in images]
    if len(ids)!=len(set(ids)):raise ValueError('Image IDs must be unique across source namespaces')
    annotations=[]
    for a in payload['annotations']:
        name=lookup[int(a['category_id'])]
        if training and name not in (new_names if new_names is not None else names):continue
        item=copy.deepcopy(a);item['original_name']=name;item['category_id']=mapping.get(name,len(names)+1)
        item.setdefault('iscrowd',0);item.setdefault('area',item['bbox'][2]*item['bbox'][3]);annotations.append(item)
    if training:
        used={int(a['image_id']) for a in annotations if not a.get('iscrowd',0)}
        images=[im for im in images if int(im['id']) in used]
        if not images:raise ValueError(f'No permitted Known training images in {source}')
    categories=[dict(id=i+1,name=name) for i,name in enumerate(names)]
    if not training:categories.append(dict(id=len(names)+1,name='unknown'))
    return dict(images=images,annotations=annotations,categories=categories)

def prepare(cfg,test_only=False):
    """Annotation lists are supplied by the benchmark, never sampled from tests."""
    root=Path(cfg['output'])/'prepared'
    previous=[]
    for task in cfg['tasks']:
        names=task['known_names'];folder=root/task['name'];folder.mkdir(parents=True,exist_ok=True)
        old=len(previous) if cfg['incremental'] else 0
        if cfg['incremental'] and names[:old]!=previous:raise ValueError('Class order must preserve the previous Known prefix')
        for key in (() if test_only else ('train_new','open_train')):
            source=source_path(cfg,task[key]);data=convert(source,names,True,names[old:] if key=='train_new' else None)
            publish(folder/(key+'.json'),data)
        if old and not test_only:
            if not task.get('replay'):raise ValueError('Task2 onward requires the official replay annotation list')
            publish(folder/'replay.json',convert(source_path(cfg,task['replay']),names,True))
        for test in task['tests']:
            publish(folder/('test_'+test['name']+'.json'),convert(source_path(cfg,test['annotations']),names))
        publish(folder/'classes.json',dict(known_names=names,previous_known=old,new_names=names[old:]))
        previous=names
    return root

class Images(Dataset):
    def __init__(self,annotation,image_root):
        self.annotation=Path(annotation);self.payload=read(annotation);self.images=self.payload['images'];self.root=Path(image_root)
        self.annotations=defaultdict(list)
        for a in self.payload['annotations']:
            if not a.get('iscrowd',0) and a['bbox'][2]>0 and a['bbox'][3]>0:self.annotations[int(a['image_id'])].append(a)
    def __len__(self):return len(self.images)
    def target(self,index):
        im=self.images[index];anns=self.annotations[int(im['id'])];w,h=im['width'],im['height']
        xy=torch.tensor([[x,y,x+bw,y+bh] for a in anns for x,y,bw,bh in [a['bbox']]],dtype=torch.float32).reshape(-1,4)
        return dict(image_id=int(im['id']),boxes=box_xyxy_to_cxcywh(xy/torch.tensor([w,h,w,h])),
            labels=torch.tensor([a['category_id']-1 for a in anns],dtype=torch.long),
            orig_size=torch.tensor([h,w]),boxes_xyxy=xy)
    def __getitem__(self,index):
        im=self.images[index];file=Path(im['file_name']);file=file if file.is_absolute() else self.root/file
        with Image.open(file) as image:value=image.convert('RGB')
        return value,self.target(index)

def image_collate(rows):return [r[0] for r in rows],[r[1] for r in rows]
def cached_collate(rows):
    return {k:torch.stack([r[0][k] for r in rows]) for k in rows[0][0]},[r[1] for r in rows]

def detector_dataset(annotation,image_root,destination):
    """Create RF-DETR's Roboflow layout from permitted training annotations."""
    destination=Path(destination);payload=read(annotation);out=copy.deepcopy(payload)
    names=[]
    for im in out['images']:
        source=Path(im['file_name']);source=source if source.is_absolute() else Path(image_root)/source
        if not source.is_file():raise FileNotFoundError(source)
        file=f"{im['id']}{source.suffix.lower()}";names.append((source.resolve(),file));im['file_name']=file
    # A fixed small Known training slice supplies the API-required monitor.
    # Fixed-budget final weights are used; this is not held-out evaluation.
    monitor_ids={int(im['id']) for im in out['images'][:min(64,len(out['images']))]}
    for split in ('train','valid','test'):
        folder=destination/split;folder.mkdir(parents=True,exist_ok=True)
        selected=out if split=='train' else dict(out,images=[im for im in out['images'] if int(im['id']) in monitor_ids],
            annotations=[a for a in out['annotations'] if int(a['image_id']) in monitor_ids])
        allowed={im['file_name'] for im in selected['images']}
        for source,name in names:
            target=folder/name
            if name in allowed and not target.exists():target.symlink_to(source)
        publish(folder/'_annotations.coco.json',selected)
    return destination
