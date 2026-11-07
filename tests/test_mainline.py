import unittest
import torch
from openproof.reference import ReferenceBank
from openproof.model import OpenProof
from openproof.config import runtime
from openproof.losses import Criterion,SEVEN_KEYS
from openproof.relations import relation_odds_probability
from openproof.episodes import meta_gradients

class Mainline(unittest.TestCase):
    def test_full_scores_losses_and_meta_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(14022)
        model=OpenProof(runtime(3,'cpu'));ref=model.decoder.reference
        x=torch.randn(36,256);y=torch.arange(3).repeat_interleave(12);ids=torch.arange(36)
        ref.fit_references(x,y,ids);banks=1-ids.remainder(2)
        distances=ref.raw_normalized_distances(x,banks);ref.fit_quantiles(distances,y,ids)
        model.train();n=12
        cached=dict(query_summary=torch.randn(2,n,256),context_summary=torch.randn(2,n,256),
            box_first=torch.rand(2,n,4)*.5+.2,box_final=torch.rand(2,n,4)*.5+.2,
            object_trajectory_features=torch.rand(2,n,26),native_logits=torch.randn(2,n,4))
        targets=[dict(image_id=i,boxes=cached['box_final'][i,:1].clone(),labels=torch.tensor([i])) for i in range(2)]
        model.prepare_batch(targets);out=model.perceive_cached(cached)
        torch.testing.assert_close(out['scores'].sum(-1),out['native_candidate_ranking_object_probability'])
        self.assertTrue(bool((out['scores']>=0).all()))
        criterion=Criterion(model.cfg);criterion.set_class_balance_counts([1,1,1])
        losses,_=criterion(model,out,targets);self.assertEqual(set(losses),set(SEVEN_KEYS))
        total=sum(losses.values());self.assertTrue(bool(torch.isfinite(total)));total.backward()
        self.assertTrue(all(p.grad is None for p in ref.parameters()))
        optimizer=torch.optim.AdamW(model.optimizer_parameter_groups(),weight_decay=1e-4)
        gradients,info=meta_gradients(model,criterion,cached,targets,optimizer,0,3)
        self.assertEqual(info['episode_active'],1);self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_unrelated_boxes_do_not_compete(self):
        boxes=torch.tensor([[[.1,.1,.1,.1],[.9,.9,.1,.1]]]);rank=torch.randn(1,2)
        unique,_,_=relation_odds_probability(torch.full((1,2,2),20.),rank,boxes)
        torch.testing.assert_close(unique,torch.ones_like(unique))

    def test_incremental_head_semantics_and_label_mask(self):
        from openproof.inheritance import expand_detector_state
        from openproof.rf_training import mask_old_logits
        old={'class_embed.weight':torch.tensor([[1.],[2.],[9.]])}
        fresh={'class_embed.weight':torch.tensor([[0.],[0.],[3.],[4.],[0.]])}
        expanded,_=expand_detector_state(old,fresh,['a','b'],['a','b','c','d'])
        torch.testing.assert_close(expanded['class_embed.weight'],torch.tensor([[1.],[2.],[3.],[4.],[9.]]))
        logits=torch.randn(1,2,5,requires_grad=True)
        masked=mask_old_logits({'pred_logits':logits},2)['pred_logits']
        masked.sum().backward()
        torch.testing.assert_close(logits.grad[...,:2],torch.zeros_like(logits.grad[...,:2]))
        torch.testing.assert_close(logits.grad[...,2:],torch.ones_like(logits.grad[...,2:]))

    def test_author_unknown_ap11_and_prediction_budget(self):
        from openproof.testing import select,unknown_author
        boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
        scores=torch.tensor([[.9,.1],[.1,.8]])
        predictions=select(boxes,scores,1,3);self.assertEqual(len(predictions),3)
        data=dict(images=[dict(id=1)],annotations=[dict(image_id=1,category_id=1,bbox=[0,0,10,10]),dict(image_id=1,category_id=1,bbox=[40,40,10,10])])
        data['annotations'][0]['category_id']=2;data['annotations'][1]['category_id']=2
        result=unknown_author(data,[dict(image_id=1,label=1,box=[0,0,10,10],score=.9)],1,expected=1)
        self.assertAlmostEqual(result['U_AP50'],6/11)

    def test_owod_author_matching_and_open_set_errors(self):
        from pathlib import Path
        import tempfile
        from openproof.config import publish
        from openproof.testing import owod_ap
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);xml=root/'xml';xml.mkdir()
            objects=''.join(f'<object><name>{name}</name><difficult>{difficult}</difficult><bndbox><xmin>{x}</xmin><ymin>1</ymin><xmax>{x+9}</xmax><ymax>10</ymax></bndbox></object>' for name,difficult,x in [('cat',0,1),('future',0,21),('cat',1,41)])
            (xml/'sample.xml').write_text('<annotation>'+objects+'</annotation>')
            cfg=dict(output=str(root/'run'),data_root=str(root));task=dict(name='task1',known_names=['cat'])
            publish(root/'run/prepared/task1/classes.json',dict(previous_known=0))
            data=dict(images=[dict(id=123,owod_id='sample',file_name='sample.jpg')])
            pred=[dict(image_id=123,label=0,score=.9,box=[0,0,10,10]),dict(image_id=123,label=0,score=.8,box=[40,0,50,10]),dict(image_id=123,label=0,score=.7,box=[20,0,30,10]),dict(image_id=123,label=1,score=.9,box=[20,0,30,10])]
            result=owod_ap(cfg,task,dict(voc_xml='xml',unknown=True),data,pred,root/'result')
            self.assertAlmostEqual(result['K_AP50'],1.)
            self.assertAlmostEqual(result['U_AP50'],1.)
            self.assertAlmostEqual(result['U_Recall'],1.)
            self.assertEqual(result['A_OSE'],1)
            self.assertAlmostEqual(result['H'],1.)

if __name__=='__main__':unittest.main()
