"""One train/test CLI for independent OSOD splits and incremental OWOD tasks."""
import argparse
import json
from pathlib import Path
from .config import load,publish

def main():
    p=argparse.ArgumentParser(prog='openproof')
    p.add_argument('action',choices=['prepare','train','test','score','run','summary'])
    p.add_argument('--config',required=True)
    p.add_argument('--task',help='Independent split, or the last task of an incremental prefix')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    a=p.parse_args();cfg=load(a.config)
    from .data import prepare
    if a.action=='prepare':prepare(cfg);return
    if a.action=='summary':return summary(cfg)
    if a.action!='score':prepare(cfg,test_only=a.action=='test')
    tasks=cfg['tasks']
    if a.task:
        index=next(i for i,t in enumerate(tasks) if t['name']==a.task)
        tasks=tasks[:index+1] if cfg['incremental'] and a.action not in ('test','score') else [tasks[index]]
    previous=None
    for task in tasks:
        if a.action in ('train','run'):
            from .rf_training import train as train_rf
            from .training import train as train_open
            train_rf(cfg,task,previous,a.device)
            import gc,torch
            gc.collect()
            if a.device=='cuda':torch.cuda.empty_cache()
            train_open(cfg,task,previous,a.device)
            gc.collect()
            if a.device=='cuda':torch.cuda.empty_cache()
        if a.action in ('test','score','run'):
            from .testing import predict,score
            for test in task['tests']:
                if a.action!='score':predict(cfg,task,test,a.device)
                score(cfg,task,test)
        if cfg['incremental']:previous=Path(cfg['output'])/task['name']
    summary(cfg)

def summary(cfg):
    rows=[]
    for task in cfg['tasks']:
        for test in task['tests']:
            path=Path(cfg['output'])/task['name']/'tests'/test['name']/'metrics.json'
            if path.is_file():rows.append(json.loads(path.read_text()))
    publish(Path(cfg['output'])/'results.json',dict(rows=rows))
    for row in rows:print(json.dumps(row,ensure_ascii=False))

if __name__=='__main__':main()
