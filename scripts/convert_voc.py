"""Convert an official VOC/OWOD image list to COCO without collapsing IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image-list',required=True);p.add_argument('--xml-root',required=True)
    p.add_argument('--images-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();names=Path(a.image_list).read_text().splitlines();images=[];annotations=[];categories={};seen=set()
    for name in names:
        name=name.strip()
        if not name:continue
        file=Path(a.images_root)/(name+'.jpg')
        if not file.is_file():raise FileNotFoundError(file)
        key=str(file.resolve())
        if key in seen:continue
        seen.add(key);root=ET.parse(Path(a.xml_root)/(name+'.xml')).getroot()
        # Names such as 000139 and 000000000139 remain different identities.
        image_id=int.from_bytes(hashlib.sha256(name.encode()).digest()[:6],'big')
        size=root.find('size');images.append(dict(id=image_id,file_name=name+'.jpg',owod_id=name,
            width=int(size.findtext('width')),height=int(size.findtext('height'))))
        for obj in root.findall('object'):
            label=obj.findtext('name');category=categories.setdefault(label,len(categories)+1)
            box=obj.find('bndbox');x,y,x2,y2=[int(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
            annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=category,
                bbox=[x-1,y-1,x2-x+1,y2-y+1],area=(x2-x+1)*(y2-y+1),iscrowd=0,
                difficult=int(obj.findtext('difficult','0')),voc_box=[x,y,x2,y2]))
    if len(images)!=len({im['id'] for im in images}):raise RuntimeError('Image identity collision')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(images=images,annotations=annotations,categories=[dict(id=i,name=n) for n,i in categories.items()])))
    print(f'{len(images)} unique images; {len(annotations)} objects -> {out}')

if __name__=='__main__':main()
