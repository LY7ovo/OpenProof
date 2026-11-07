"""Protocol distinctions that perfect-recall fixtures cannot detect."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from openproof.config import publish,read
from openproof.testing import coco_ap,owod_ap


class MetricProtocols(unittest.TestCase):
    def test_coco101_partial_recall_for_known_and_unknown(self):
        data=dict(images=[dict(id=1),dict(id=2)],categories=[dict(id=1,name='cat'),dict(id=2,name='unknown')],
                  annotations=[dict(id=i*2+c,image_id=i,category_id=c,bbox=[c*20,0,10,10],area=100,iscrowd=0)
                               for i in (1,2) for c in (1,2)])
        pred=[dict(image_id=1,label=c-1,box=[c*20,0,c*20+10,10],score=.9) for c in (1,2)]
        result=coco_ap(data,pred,1,100)
        # One of two GTs recalled perfectly: AP101=51/101, AP11=6/11,
        # continuous=.5. This catches changing only the metric's label.
        for key in ('K_AP','K_AP50','U_AP','U_AP50'):
            self.assertAlmostEqual(result[key],51/101)
        self.assertAlmostEqual(result['U_Recall'],.5)

    def test_coco_iou_average_is_distinct_from_ap50(self):
        data=dict(images=[dict(id=1)],categories=[dict(id=1,name='cat'),dict(id=2,name='unknown')],
                  annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0)])
        pred=[dict(image_id=1,label=0,box=[0,0,6.2,10],score=.9)]
        result=coco_ap(data,pred,1,100)
        self.assertAlmostEqual(result['K_AP50'],1.)
        self.assertAlmostEqual(result['K_AP'],.3)  # passes .50/.55/.60 only
        self.assertIsNone(result['U_AP'])

    def test_uod_cpu_score_cli_uses_mixed_protocol_and_saved_predictions(self):
        import yaml
        from openproof.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'run';folder=output/'voc/tests/mixed'
            task=dict(name='voc',known_names=['cat'],tests=[dict(name='mixed',protocol='uod',top_k=500,
                      unknown=True,expected_unknown_images=2)])
            cfg=dict(benchmark='voccoco',data_root=str(root),output=str(output),pretrained=str(root/'missing.pth'),
                     incremental=False,tasks=[task])
            data=dict(images=[dict(id=i) for i in range(1,5)],categories=[dict(id=1,name='cat'),dict(id=2,name='unknown')],
                      annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
                                   dict(id=2,image_id=2,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
                                   dict(id=3,image_id=1,category_id=2,bbox=[20,0,10,10],area=100,iscrowd=0),
                                   dict(id=4,image_id=3,category_id=2,bbox=[20,0,10,10],area=100,iscrowd=0)])
            pred=[dict(image_id=1,label=0,box=[0,0,10,10],score=.9),
                  dict(image_id=1,label=1,box=[20,0,30,10],score=.9),
                  dict(image_id=4,label=1,box=[0,0,10,10],score=.99)]
            publish(output/'prepared/voc/test_mixed.json',data)
            publish(folder/'manifest.json',dict(complete=True,images=4,predictions=3))
            shard=folder/'shards/00000.json';publish(shard,dict(ids=[1,2,3,4],predictions=pred))
            before=shard.read_bytes()
            publish(folder/'metrics.json',dict(AP_interpolation=11,old_result=True))
            old=(folder/'metrics.json').read_bytes()
            config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg))
            # There are no images, RF weights, OPEN weights or train annotations:
            # this CLI can succeed only by consuming the saved prediction shards.
            with patch.object(sys,'argv',['openproof','score','--config',str(config),'--device','cpu']):main()
            result=read(folder/'metrics.json');m=result['metrics']
            self.assertAlmostEqual(m['K_AP'],51/101)
            self.assertAlmostEqual(m['K_AP50'],51/101)
            self.assertAlmostEqual(m['U_AP50'],6/11)
            self.assertAlmostEqual(m['U_Recall'],.5)
            self.assertIsNone(m['U_AP'])
            self.assertEqual(m['unknown_images'],2)
            self.assertEqual(result['AP_protocol']['Known']['interpolation'],'COCO101')
            self.assertEqual(result['AP_protocol']['Unknown']['interpolation'],'VOC07 AP11')
            self.assertEqual(shard.read_bytes(),before)
            self.assertEqual((folder/'metrics_before_protocol_fix.json').read_bytes(),old)

    def test_owod_continuous_partial_recall_and_final_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);xml=root/'xml';xml.mkdir()
            objects=''.join(f'<object><name>{name}</name><difficult>0</difficult><bndbox><xmin>{x}</xmin><ymin>1</ymin><xmax>{x+9}</xmax><ymax>10</ymax></bndbox></object>'
                            for name,x in [('cat',1),('future',21),('cat',41),('future',61)])
            (xml/'sample.xml').write_text('<annotation>'+objects+'</annotation>')
            cfg=dict(output=str(root/'run'),data_root=str(root));task=dict(name='task1',known_names=['cat'])
            publish(root/'run/prepared/task1/classes.json',dict(previous_known=0))
            data=dict(images=[dict(id=123,owod_id='sample',file_name='sample.jpg')])
            pred=[dict(image_id=123,label=0,score=.9,box=[0,0,10,10]),dict(image_id=123,label=1,score=.9,box=[20,0,30,10])]
            result=owod_ap(cfg,task,dict(voc_xml='xml',unknown=True),data,pred,root/'result')
            self.assertAlmostEqual(result['K_AP50'],.5)
            self.assertAlmostEqual(result['U_AP50'],.5)
            self.assertAlmostEqual(result['U_Recall'],.5)
            task['known_names']=['cat','future']
            publish(root/'run/prepared/task1/classes.json',dict(previous_known=1))
            result=owod_ap(cfg,task,dict(voc_xml='xml',unknown=False),data,pred,root/'final')
            self.assertAlmostEqual(result['previous_K_AP50'],.5)
            self.assertAlmostEqual(result['new_K_AP50'],.5)
            for key in ('U_AP50','U_Recall','H','WI_08'):self.assertIsNone(result[key])


if __name__=='__main__':unittest.main()
