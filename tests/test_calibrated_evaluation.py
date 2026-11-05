"""Readout arithmetic/order and fail-closed trajectory reuse before test scoring."""
import copy
import io
import unittest
import numpy as np
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,cohen_kappa_score,confusion_matrix
from evaluation.calibrated_baseline import compact_readout,score_compact,require_reuse_receipts,scheduler_reuse


class CalibratedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([1,1,1,2,2,3,3])
        self.q=torch.tensor([[.8,.1,.1],[.6,.2,.2],[.1,.8,.1],[.8,.1,.1],[.1,.8,.1],[.1,.2,.7],[.1,.7,.2]])

    def test_metrics_match_independent_library_and_direct_nll(self):
        m,packet=compact_readout(self.y,self.q,3);pred=self.q.argmax(1).numpy()+1
        self.assertEqual(m['confusion_matrix'],confusion_matrix(self.y,pred,labels=[1,2,3]).tolist())
        self.assertEqual(m['oa'],accuracy_score(self.y,pred))
        self.assertAlmostEqual(m['aa'],balanced_accuracy_score(self.y,pred))
        self.assertAlmostEqual(m['kappa'],cohen_kappa_score(self.y,pred))
        self.assertAlmostEqual(m['nll'],float(-self.q[torch.arange(7),self.y-1].log().mean()))
        self.assertEqual(packet['prediction'].dtype,torch.int16)

    def test_unsorted_split_order_and_compact_serialization(self):
        m,_=compact_readout(self.y,self.q,3);order=np.array([5,0,6,2,4,1,3])
        candidate,packet=compact_readout(self.y[order],self.q[order],3)
        for key in ('confusion_matrix','per_class','oa','aa','kappa'):self.assertEqual(m[key],candidate[key])
        stream=io.BytesIO();torch.save(packet,stream);stream.seek(0)
        restored=torch.load(stream,weights_only=True)
        self.assertTrue(torch.equal(restored['prediction'],self.q[order].argmax(1)+1))
        self.assertEqual(score_compact(self.y[order],restored,3),candidate)

    def test_invalid_probabilities_labels_shapes_and_missing_classes_rejected(self):
        for y,q in ((self.y,self.q*float('nan')),(self.y,-self.q),(self.y,self.q*10),
                    (self.y[:2],self.q),(self.y[:,None],self.q),(self.y+.1,self.q),
                    (np.zeros(7),self.q),(np.ones(7),self.q)):
            with self.assertRaises(ValueError):compact_readout(y,q,3)

    def test_compact_nll_zero_clamp_and_count_protection(self):
        m,packet=compact_readout(np.array([1,2]),torch.tensor([[0.,1.],[.2,.8]]),2)
        self.assertAlmostEqual(m['nll'],float(-(torch.tensor([1e-8,.8])).log().mean()))
        packet['true_probability']=torch.tensor([.5])
        with self.assertRaises(ValueError):score_compact(np.array([1,2]),packet,2)

    def test_receipts_require_both_devices_exact_code_all_seeds_and_pass(self):
        code={'commit':'fixed','source_hash':'fixed','dirty':False}
        receipts={d:{'status':'passed','code':code,'device':d,'seeds':[64,512,2048],
            'raw_training_unchanged':True,'rng_unchanged':True,'real_optimizer_updates':0} for d in ('cpu','cuda')}
        require_reuse_receipts(receipts,code)
        with self.assertRaises(ValueError):require_reuse_receipts({'cpu':receipts['cpu']},code)
        for key,value in [('status','failed'),('seeds',[64]),('code',{'commit':'wrong'}),
                          ('rng_unchanged',False),('real_optimizer_updates',1)]:
            bad=copy.deepcopy(receipts);bad['cuda'][key]=value
            with self.assertRaises(ValueError):require_reuse_receipts(bad,code)

    def scheduler_fixture(self):
        settings={'config':{'training':{'lr':.01,'head_lr':.02}},'scheduler':{'factor':.5,'patience':0,'threshold':.0001}}
        optimizer=torch.optim.Adam([{'params':[torch.nn.Parameter(torch.zeros(1))],'lr':.01},
                                    {'params':[torch.nn.Parameter(torch.zeros(1))],'lr':.02}])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=0,threshold=.0001)
        history=[{'step':3,'val_loss':1.},{'step':33,'val_loss':1.1}];states=[]
        for event in history:
            scheduler.step(event['val_loss']);states.append({'cursor':event['step'],'scheduler':copy.deepcopy(scheduler.state_dict())})
        return settings,history,states

    def test_scheduler_replay_authenticates_original_state_and_accepts_equal_lr(self):
        settings,history,states=self.scheduler_fixture()
        rows=scheduler_reuse(history,history,states,settings)
        self.assertEqual(rows[-1]['raw_lr'],[.005,.01]);self.assertEqual(rows[-1]['raw_lr'],rows[-1]['calibrated_lr'])
        states[-1]['scheduler']['best']=999.
        with self.assertRaisesRegex(ValueError,'did not reproduce'):scheduler_reuse(history,history,states,settings)

    def test_scheduler_lr_divergence_rejects_reuse(self):
        settings,history,states=self.scheduler_fixture();calibrated=copy.deepcopy(history);calibrated[-1]['val_loss']=.9
        with self.assertRaisesRegex(ValueError,'changes LR'):scheduler_reuse(history,calibrated,states,settings)

    def test_scheduler_incomplete_nonfinite_or_wrong_cursor_rejected(self):
        settings,history,states=self.scheduler_fixture()
        with self.assertRaises(ValueError):scheduler_reuse(history,history[:1],states,settings)
        bad=copy.deepcopy(history);bad[-1]['val_loss']=float('nan')
        with self.assertRaises(ValueError):scheduler_reuse(history,bad,states,settings)
        states[-1]['cursor']=34
        with self.assertRaises(ValueError):scheduler_reuse(history,history,states,settings)


if __name__=='__main__':unittest.main()
