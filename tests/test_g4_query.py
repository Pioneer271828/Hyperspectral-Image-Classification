import copy,json,math,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
import torch
from evaluation import g4_target as g4
from evaluation.classification_metrics import metrics_from_confusion
from joint_baseline import NAMES
from joint_training import schedule_for,run_training
from loadData.dataset_registry import DATASETS,file_hash
from loadData.target_label_protocol import make_target_family
from model.research_config import digest
from research_runtime import write_json
from target_initialization import initialize_target,state_id,METHODS
from target_study import full_settings,case_name,EXPECTED,TargetStudyRuntime
from test_target_training import fixture

DEVICE=os.environ.get('G4I_DEVICE','cpu');torch.set_num_threads(1)


def frozen_fixture(root,*,selected=1500,updates=1500,producer_commit=g4.TRAINING_COMMIT):
    """Synthetic metadata, not trained weights; native audits are explicitly mocked."""
    train=root/'training';(train/'cases').mkdir(parents=True);(train/'reviews').mkdir()
    files={'preserved.py':'a'*64};producer={'commit':producer_commit,'branch':'fixture','dirty':False,
        'files':files,'source_hash':digest(files)}
    current={**producer,'commit':'b'*40,'files':{**files,'additive.py':'c'*64}}
    current['source_hash']=digest(current['files']);cases=[];models=[]
    for target in NAMES:
        graph=root/(target+'-unlabeled.pt');graph.write_bytes(target.encode())
        for seed in (64,512,2048):
            for method in METHODS:
                name=case_name(target,seed,method);folder=train/'cases'/name;folder.mkdir()
                settings=full_settings(target,method,seed);provenance={'protected_files':{str(graph):file_hash(graph)},
                    'family_hash':digest({'target':target})}
                case={'name':name,'target':target,'seed':seed,'method':method,'settings':settings,'out':str(folder),
                    'inputs':{'target':target},'source_spec':{} if method==METHODS[0] else {'fixture':True},
                    'source_result':None if method==METHODS[0] else {'fixture':True},'provenance':provenance,
                    'initial_state_sha256':'d'*64}
                schedule=schedule_for(settings,task_names=(target,))
                identity={'protocol':settings['protocol'],'settings':settings,'settings_hash':digest(settings),
                    'execution_device':'cuda','consumer_source_hash':producer['source_hash'],'prepared':provenance,
                    'schedule_sha256':digest(schedule)}
                write_json(folder/'identity.json',identity)
                for filename in ('primary.pt','primary.pt.json','last.pt','last.pt.json','selected-probabilities.pt',
                                 'state-001500.pt','raw-validation-001500.pt'):(folder/filename).write_bytes(filename.encode())
                write_json(folder/'progress.json',{'status':'completed','cursor':1500,'state_file':'state-001500.pt',
                    'state_sha256':file_hash(folder/'state-001500.pt')})
                write_json(folder/'trace.json',[{'step':i,'dataset':target} for i in range(1,1501)])
                steps=list(range(1,1472,30))+[1500]
                write_json(folder/'history.json',[{'step':s,'val_loss':3-i*.01,'selection_score':-(3-i*.01)} for i,s in enumerate(steps)])
                artifacts={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
                result={'status':'completed','optimizer_steps':updates,'start_code':producer,'settings':settings,
                    'test_scoring':False,'selected_step':selected,'per_dataset_updates':{target:1500},
                    'identity':identity,'schedule_sha256':digest(schedule),'artifacts':artifacts}
                write_json(folder/'result.json',result)
                review=train/'reviews'/(name+'.json')
                write_json(review,{'status':'completed','updates':1500,'selected_step':selected,
                    'artifacts':{**artifacts,'result.json':file_hash(folder/'result.json')},'completed_resume_updates':0})
                models.append({'name':name,'selected_step':selected,'primary_sha256':file_hash(folder/'primary.pt'),
                    'sidecar_sha256':file_hash(folder/'primary.pt.json'),'result_sha256':file_hash(folder/'result.json'),
                    'review_sha256':file_hash(review)})
                cases.append(case)
    launch={'code':producer,'study':copy.deepcopy(EXPECTED),'out_dir':str(train),'cases':cases,
        'query_labels_accessed':False,'test_scoring':False,'expected_updates':40500}
    write_json(train/'launch.json',launch);lh=file_hash(train/'launch.json')
    write_json(train/'training-acceptance.json',{'status':'passed','source':producer,'launch_sha256':lh,
        'optimizer_updates':40500,'query_labels_accessed':False,'test_scoring':False,'models':models})
    return train,lh,file_hash(train/'training-acceptance.json'),current


def native_stub(row,*,device):
    return {'name':row['case']['name'],'status':'passed','head_metrics':6,
        'class_recalls':6*DATASETS[row['case']['target']][2],'probability_max_abs_error':0.,'query_labels_read':False}


def metric_rows():
    rows=[]
    for target in NAMES:
        classes=DATASETS[target][2]
        for i,seed in enumerate((64,512,2048)):
            for method,offset in zip(METHODS,(0,2,-5)):
                correct=80+i+offset;cm=np.zeros((classes,classes),np.int64)
                for c in range(classes):cm[c,c]=correct;cm[c,(c+1)%classes]=100-correct
                metric=metrics_from_confusion(cm,range(1,classes+1));metric['nll']=1-metric['oa']
                rows.append({'target':target,'seed':seed,'method':method,'metrics':{h:copy.deepcopy(metric) for h in g4.HEADS}})
    return rows


class G4QueryTests(unittest.TestCase):
    def test_global_metadata_and_last_native_failure_prevent_any_label_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=frozen_fixture(root);reader=Mock()
            stage=g4.verify_training(*args);self.assertEqual(len(stage['rows']),27)
            def fail_last(row,*,device):
                if row['case']['name']==stage['rows'][-1]['case']['name']:raise ValueError('last native audit failed')
                return native_stub(row,device=device)
            with patch.object(g4,'audit_native_case',side_effect=fail_last):
                with self.assertRaisesRegex(ValueError,'last native'):
                    _,freeze=g4.freeze_for_query(*args,root/'blocked',device=DEVICE)
                    reader(freeze)
            reader.assert_not_called();self.assertFalse(json.loads((root/'blocked/prequery-failure.json').read_text())['query_labels_read'])
            with patch.object(g4,'audit_native_case',side_effect=native_stub) as native:
                _,freeze=g4.freeze_for_query(*args,root/'allowed',device=DEVICE);g4.require_frozen_boundary(freeze);reader(freeze)
                self.assertEqual(native.call_count,27)
            self.assertEqual(reader.call_count,1);self.assertEqual(freeze['head_metric_cases'],162);self.assertEqual(freeze['class_recalls'],2160)
            before=file_hash(root/'allowed/freeze.json')
            with self.assertRaises(FileExistsError):g4.freeze_for_query(*args,root/'allowed',device=DEVICE)
            self.assertEqual(before,file_hash(root/'allowed/freeze.json'))

    def test_incomplete_budget_selection_producer_and_artifact_tampering_block_before_native(self):
        for key,value in [('updates',1499),('selected',1),('producer_commit','0'*40)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                args=frozen_fixture(Path(tmp),**{key:value})
                with patch.object(g4,'audit_native_case') as native:
                    with self.assertRaises(ValueError):g4.freeze_for_query(*args,Path(tmp)/'out',device=DEVICE)
                    native.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=frozen_fixture(root);train,lh,ah,current=args
            case=json.loads((train/'launch.json').read_text())['cases'][0]
            for key in ('primary.pt','primary.pt.json','raw-validation-001500.pt','selected-probabilities.pt'):
                path=Path(case['out'])/key;before=path.read_bytes();path.write_bytes(before+b'tampered')
                with self.assertRaises(ValueError):g4.verify_training(*args)
                path.write_bytes(before)
            graph=Path(next(iter(case['provenance']['protected_files'])));before=graph.read_bytes();graph.write_bytes(b'changed')
            with self.assertRaises(ValueError):g4.verify_training(*args)
            graph.write_bytes(before)
            ap=train/'training-acceptance.json';before=ap.read_bytes();ap.unlink()
            with self.assertRaises(FileNotFoundError):g4.verify_training(*args)
            ap.write_bytes(before)
            bad=copy.deepcopy(current);bad['files']['preserved.py']='f'*64;bad['source_hash']=digest(bad['files'])
            with self.assertRaises(ValueError):g4.verify_training(train,lh,ah,bad)

    def test_actual_weight_family_and_raw_gt_hash_binding_after_global_barrier(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=frozen_fixture(root)
            with patch.object(g4,'audit_native_case',side_effect=native_stub):stage,freeze=g4.freeze_for_query(*args,root/'audit',device=DEVICE)
            case=copy.deepcopy(stage['rows'][0]['case']);target=case['target']
            labels=(np.arange(400)%9+1).reshape(20,20);family=make_target_family(labels,target,64)
            fp=root/'family.json';write_json(fp,family);raw=root/'labels.mat';savemat(raw,{'paviaU_gt':labels})
            case['inputs'].update(family=str(fp),family_sha256=file_hash(fp));case['provenance']['family_hash']=family['family_hash']
            # Explicit synthetic barrier fixture; no trained-model claim.
            model=next(m for m in freeze['models'] if m['name']==case['name'])
            model.update(case_sha256=digest(case),family_hash=family['family_hash'])
            cohort=root/'cohort.json';write_json(cohort,{'protocol':'acc1p_joint_prepared_v1','datasets':{target:{'raw':{
                'class_count':9,'label_path':'labels.mat','label_sha256':file_hash(raw)}}}})
            ch=file_hash(cohort)
            with patch.object(g4,'PREPARATION_HASH',ch):
                release,pins=g4.read_query_labels(freeze,case,root,cohort,ch)
                self.assertEqual(release['checkpoint_sha256'],file_hash(Path(case['out'])/'primary.pt'))
                self.assertEqual(release['labels'],labels.reshape(-1)[release['indices']].tolist())
                bad=copy.deepcopy(case);bad['provenance']['family_hash']='e'*64
                with self.assertRaisesRegex(ValueError,'outside the frozen'):g4.read_query_labels(freeze,bad,root,cohort,ch)
                raw.write_bytes(b'changed')
                with patch('scipy.io.loadmat') as decode:
                    with self.assertRaisesRegex(ValueError,'raw label file'):g4.read_query_labels(freeze,case,root,cohort,ch)
                    decode.assert_not_called()
            with self.assertRaises(ValueError):g4.read_query_labels({'models':[]},case,root,cohort,ch)

    def test_independent_hand_metrics_branch_denominators_and_invalid_labels(self):
        labels=np.array([1,1,2,2,3,3]);predictions={'fused':[1,2,2,3,3,1],'cnn':[1,1,1,2,3,2],'graph':[1,3,2,2,1,3]}
        probabilities={}
        for h,ids in predictions.items():
            p=np.full((6,3),.1);p[np.arange(6),np.array(ids)-1]=.8;probabilities[h]=p
        metrics,branch=g4.query_metrics(labels,probabilities)
        self.assertEqual(metrics['fused']['oa'],.5);self.assertEqual(metrics['fused']['aa'],.5)
        self.assertAlmostEqual(metrics['fused']['kappa'],.25)
        self.assertAlmostEqual(metrics['fused']['nll'],(-math.log(.8)-math.log(.1))/2)
        self.assertEqual(branch['correction']['numerator'],2);self.assertEqual(branch['correction']['denominator'],4)
        self.assertEqual(branch['harm']['numerator'],1);self.assertEqual(branch['harm']['denominator'],2)
        self.assertEqual(branch['branch_oracle_accuracy'],1.)
        for bad in (labels*0,labels.astype(float),labels.reshape(2,3)):
            with self.assertRaises(ValueError):g4.query_metrics(bad,probabilities)

    def test_paired_sample_sd_and_class_protection_not_only_overall_gain(self):
        rows=metric_rows();summary=g4.paired_summary(rows)
        self.assertAlmostEqual(summary['macro'][METHODS[0]]['oa']['mean'],.81)
        self.assertAlmostEqual(summary['macro'][METHODS[0]]['oa']['sample_sd'],.01)
        self.assertAlmostEqual(summary['contrasts'][METHODS[1]]['paired_macro_oa_delta']['mean'],.02)
        self.assertTrue(summary['contrasts'][METHODS[1]]['passes_all_gates'])
        self.assertFalse(summary['contrasts'][METHODS[2]]['passes_all_gates'])
        for row in rows:
            if row['target']=='paviaU' and row['method']==METHODS[1]:
                cm=np.asarray(row['metrics']['fused']['confusion_matrix']);cm[0]=0;cm[0,0]=75;cm[0,1]=25
                metric=metrics_from_confusion(cm,range(1,10));metric['nll']=.1;row['metrics']['fused']=metric
        contrast=g4.paired_summary(rows)['contrasts'][METHODS[1]]
        self.assertTrue(contrast['gates']['positive_paired_macro_mean']);self.assertFalse(contrast['gates']['all_class_recall_protection'])
        with self.assertRaises(ValueError):g4.paired_summary(rows[:-1])

    def test_real_model_query_forward_is_eval_immutable_and_finite_on_cpu_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            (data,_,_),_=fixture(Path(tmp));item=data['xuzhou']
            model,_=initialize_target('xuzhou',METHODS[0],model_seed=64,head_seed=1064,device=DEVICE);model.eval()
            before=state_id(model);ids=list(range(200,209))
            probabilities=g4.predict_query(model,item,ids)
            self.assertEqual(before,state_id(model));self.assertTrue(all(not p.requires_grad for p in probabilities.values()))
            self.assertTrue(all(p.shape==(9,9) for p in probabilities.values()))
            metrics,_=g4.query_metrics(np.arange(1,10),{h:p.numpy() for h,p in probabilities.items()})
            self.assertTrue(all(math.isfinite(m['nll']) for m in metrics.values()))
            with self.assertRaises(ValueError):g4.predict_query(model,item,[0,0])

    def test_actual1500update_tiny_model_native_state_and_trainval_audit(self):
        # Real1500-step execution on a tiny synthetic scene. It does not reproduce
        # a real dataset or the27-case accuracy campaign.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,_),paths=fixture(root);out=root/'native'
            opts=full_settings('xuzhou',METHODS[0],64,device=DEVICE)
            initial_model,initial=initialize_target('xuzhou',METHODS[0],model_seed=64,head_seed=1064,device=DEVICE)
            del initial_model
            source={'commit':'fixture','files':{},'source_hash':digest({})}
            runtime=TargetStudyRuntime(opts,data,out,source_spec={},initialization_state_sha256=initial['initial_state_sha256'])
            result=run_training(opts,data,out,provenance,source,device=DEVICE,runtime_adapter=runtime)['result']
            case={'name':'fixture-native','target':'xuzhou','seed':64,'method':METHODS[0],'settings':opts,
                'out':str(out),'source_spec':{},'source_result':None,'initial_state_sha256':initial['initial_state_sha256'],
                'provenance':provenance,'inputs':{'observation':str(paths[0]),'record':str(paths[1]),
                    'record_sha256':file_hash(paths[1]),'family':str(paths[2]),'family_sha256':file_hash(paths[2])}}
            row={'case':case,'result':result}
            receipt=g4.audit_native_case(row,device=DEVICE)
            self.assertEqual(receipt['head_metrics'],6);self.assertEqual(receipt['class_recalls'],54)
            state_file=out/json.loads((out/'progress.json').read_text())['state_file']
            state=torch.load(state_file,map_location='cpu',weights_only=True);state['best_score']+=.1
            torch.save(state,state_file)
            with self.assertRaisesRegex(ValueError,'Authoritative training state'):g4.audit_native_case(row,device=DEVICE)

    def test_completed_evaluation_is_readonly_and_keeps_scientific_status(self):
        from scripts.evaluate_target_full import completed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model.pt';model.write_bytes(b'weights')
            raw=root/'raw-labels.mat';raw.write_bytes(b'labels');out=root/'evaluation';out.mkdir()
            pred=out/'pred.pt';pred.write_bytes(b'predictions');identity={'fixture':True}
            write_json(out/'result.json',{'status':'completed','identity':identity,'policy':g4.POLICY,
                'protected_training_files':{str(model):file_hash(model)},'raw_label_pins':{str(raw):file_hash(raw)},
                'artifacts':{'pred.pt':file_hash(pred)}})
            before={p.name:file_hash(p) for p in out.iterdir()}
            value=completed(out,identity)
            self.assertEqual(value['operation'],'verified_completed');self.assertEqual(value['result']['status'],'completed')
            self.assertEqual(value['evaluation_updates_this_call'],0)
            self.assertEqual(before,{p.name:file_hash(p) for p in out.iterdir()})
            pred.write_bytes(b'tampered')
            with self.assertRaises(ValueError):completed(out,identity)


if __name__=='__main__':unittest.main()
