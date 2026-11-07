"""Obtain RF-DETR Small's official pretrained initialization separately."""
from pathlib import Path
import torch
from rfdetr import RFDETRSmall

def main():
    destination=Path(__file__).resolve().parents[1]/'weights/rf-detr-small.pth'
    if destination.exists():
        print(destination);return
    api=RFDETRSmall(device='cpu',resolution=512)
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=api.model.model.state_dict(),args=vars(api.model.args),
        model_config=api.model_config.model_dump(),model_name='rf-detr-small',rfdetr_version='1.9.3'),destination)
    print(destination)

if __name__=='__main__':main()
