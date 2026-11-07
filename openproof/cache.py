"""Disk-backed FP32 detector features; no learned OPEN output is cached."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data import Images,image_collate
from .config import read,publish

def build(model,dataset,folder,batch_size=16):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    import hashlib
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):h.update(block)
        return h.hexdigest()
    identity=dict(annotation=digest(dataset.annotation),RF_checkpoint=digest(model.rf.checkpoint),geometry=model.rf.geometry)
    identity['teacher']=[digest(path) for path in getattr(model,'_teacher_files',[])]
    marker=folder/'metadata.json'
    if marker.is_file():
        old=read(marker)
        if old['identity']!=identity:raise RuntimeError('Cache belongs to different data or detector')
        return Features(folder,dataset)
    arrays={};shapes={};offset=0
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=2,collate_fn=image_collate)
    model.eval()
    with torch.no_grad():
        for images,targets in loader:
            values=model.extract_frozen_native_cache(images)
            for key,value in values.items():
                value=value.detach().cpu().float().numpy()
                if key not in arrays:
                    shapes[key]=[len(dataset),*value.shape[1:]]
                    arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype='float32',shape=tuple(shapes[key]))
                arrays[key][offset:offset+len(images)]=value
            offset+=len(images)
            if offset%512<len(images) or offset==len(dataset):print(f'[RF cache] {offset}/{len(dataset)}',flush=True)
    for array in arrays.values():array.flush()
    publish(marker,dict(identity=identity,shapes=shapes,images=offset))
    return Features(folder,dataset)

class Features(Dataset):
    def __init__(self,folder,dataset):
        self.folder=Path(folder);self.dataset=dataset;self.metadata=read(self.folder/'metadata.json');self.arrays={}
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        if not self.arrays:self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in self.metadata['shapes']}
        return {k:torch.from_numpy(np.array(v[index],copy=True)) for k,v in self.arrays.items()},self.dataset.target(index)
    def __getstate__(self):return dict(self.__dict__,arrays={})
