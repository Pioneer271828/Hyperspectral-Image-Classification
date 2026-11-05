import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from evaluation.g4_target import query_metrics,paired_summary,POLICY,TRAINING_COMMIT
from evaluation.g4_target_verification import (oracle_metrics,oracle_branch,oracle_summary,assert_tree,verify_saved)
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS,file_hash
from loadData.target_label_protocol import make_target_family
from model.research_config import digest
from research_runtime import write_json
from target_initialization import METHODS
from target_study import case_name
from test_g4_query import metric_rows


def saved_fixture(root):
    """Synthetic saved-evaluation contract; no real model training is claimed."""
    raw=root/'raw';raw.mkdir();training=root/'training';training.mkdir();out=root/'evaluation';out.mkdir()
    producer={'commit':'a'*40,'dirty':False,'files':{'scorer.py':'b'*64},'source_hash':digest({'scorer.py':'b'*64})}
    current={**producer,'files':{**producer['files'],'verifier.py':'c'*64}};current['source_hash']=digest(current['files'])
    cases=[];models=[];native=[];pins={};raw_pins={};families={};cohort={'datasets':{}}
    for target in NAMES:
        c=DATASETS[target][2];labels=(np.arange(625)%c+1).reshape(25,25)
        path=raw/(target+'.mat');savemat(path,{DATASETS[target][1]+'_gt':labels})
        raw_pins[str(path)]=file_hash(path)
        cohort['datasets'][target]={'raw':{'label_path':path.name,'label_sha256':file_hash(path)}}
        family=make_target_family(labels,target,64);fp=root/(target+'-family.json');write_json(fp,family)
        families[target]=(family,labels);pins[str(fp)]=file_hash(fp)
        for seed in (64,512,2048):
            for method in METHODS:
                name=case_name(target,seed,method);d=training/name;d.mkdir();(d/'primary.pt').write_bytes(name.encode())
                case={'name':name,'target':target,'seed':seed,'method':method,'out':str(d),
                    'inputs':{'family':str(fp),'family_sha256':file_hash(fp)}}
                cases.append(case);pins[str(d/'primary.pt')]=file_hash(d/'primary.pt')
                models.append({'name':name,'target':target,'seed':seed,'method':method,'case_sha256':digest(case),
                    'primary_sha256':file_hash(d/'primary.pt'),'family_hash':family['family_hash']})
                native.append({'name':name,'status':'passed'})
    write_json(training/'launch.json',{'cases':cases});pins[str(training/'launch.json')]=file_hash(training/'launch.json')
    cohort_path=root/'cohort.json';write_json(cohort_path,cohort);raw_pins[str(cohort_path)]=file_hash(cohort_path)
    identity={'code':producer,'prepared_manifest':str(cohort_path),'prepared_manifest_sha256':file_hash(cohort_path),
        'training_root':str(training),'data_root':str(raw)}
    freeze={'verification_code':producer,'training_code':{'commit':TRAINING_COMMIT},'models':models,
        'native_checks':native,'head_metric_cases':162,'class_recalls':2160,'protected_files':pins}
    write_json(out/'freeze.json',freeze);rows=[]
    for case,model in zip(cases,models):
        target=case['target'];family,labels=families[target];ids=family['query_indices'];truth=labels.reshape(-1)[ids]
        count=DATASETS[target][2];method=case['method'];offset=(64,512,2048).index(case['seed']);stride=(7,11,5)[METHODS.index(method)]
        preds=truth.copy();errors=(np.arange(len(ids))+offset)%stride==0;preds[errors]=preds[errors]%count+1
        probabilities={}
        for h in ('fused','graph','cnn'):
            p=np.full((len(ids),count),.1/(count-1),dtype=np.float32);p[np.arange(len(ids)),preds-1]=.9
            probabilities[h]=p
        metrics,branch=query_metrics(truth,probabilities);payload=out/(case['name']+'-query.pt')
        with payload.open('wb') as stream:torch.save({'target':target,'seed':case['seed'],'method':method,
            'checkpoint_sha256':model['primary_sha256'],'family_hash':family['family_hash'],
            'indices':torch.tensor(ids,dtype=torch.long),'labels':torch.from_numpy(truth.astype(np.int64)),
            'probabilities':{h:torch.from_numpy(p) for h,p in probabilities.items()},'freeze_sha256':file_hash(out/'freeze.json')},stream)
        row={**{k:case[k] for k in ('name','target','seed','method')},'metrics':metrics,'branch':branch,
            'checkpoint_sha256':model['primary_sha256'],'family_hash':family['family_hash'],
            'query_payload':payload.name,'query_payload_sha256':file_hash(payload)}
        rows.append(row);write_json(out/(case['name']+'-metrics.json'),row)
    summary=paired_summary(rows);write_json(out/'summary.json',summary)
    result={'status':'completed','policy':POLICY,'identity':identity,'rows':rows,'summary':summary,
        'protected_training_files':pins,'raw_label_pins':raw_pins,
        'artifacts':{p.name:file_hash(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'result.json',result)
    return out,file_hash(out/'result.json'),current


class G4IndependentVerificationTests(unittest.TestCase):
    def test_independent_metrics_branches_and_paired_sample_statistics(self):
        labels=np.array([1,2,3,1,2,3]);p=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7],[.1,.8,.1],[.1,.8,.1],[.7,.1,.2]])
        probabilities={'fused':p,'graph':p[:,[1,2,0]],'cnn':p[:,[2,0,1]]}
        metrics,branch=query_metrics(labels,probabilities)
        for h in probabilities:assert_tree(oracle_metrics(labels,probabilities[h]),metrics[h])
        assert_tree(oracle_branch(labels,probabilities),branch)
        rows=metric_rows();assert_tree(oracle_summary(rows),paired_summary(rows))
        bad=copy.deepcopy(paired_summary(rows));bad['macro'][METHODS[0]]['oa']['sample_sd']=0
        with self.assertRaises(ValueError):assert_tree(oracle_summary(rows),bad)

    def test_all27_saved_truth_predictions_summary_and_files_are_independently_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,sha,current=saved_fixture(Path(tmp));before={p.name:file_hash(p) for p in out.iterdir()}
            receipt=verify_saved(out,sha,current)
            self.assertEqual(receipt['head_metric_cases'],81);self.assertEqual(receipt['class_recalls'],1080)
            self.assertEqual(receipt['branch_records'],27);self.assertEqual(receipt['class_branch_records'],360)
            self.assertEqual(receipt['analysis_optimizer_updates'],0)
            self.assertEqual(before,{p.name:file_hash(p) for p in out.iterdir()})
            # Self-consistent artifact hashes cannot conceal wrong saved truth.
            result=json.loads((out/'result.json').read_text());row=result['rows'][0];payload_path=out/row['query_payload']
            payload=torch.load(payload_path,weights_only=True);payload['labels'][0]=payload['labels'][0]%9+1
            with payload_path.open('wb') as stream:torch.save(payload,stream)
            row['query_payload_sha256']=file_hash(payload_path)
            metric_path=out/(row['name']+'-metrics.json');write_json(metric_path,row)
            result['artifacts'][payload_path.name]=file_hash(payload_path);result['artifacts'][metric_path.name]=file_hash(metric_path)
            write_json(out/'result.json',result)
            with self.assertRaisesRegex(ValueError,'Saved query truth'):verify_saved(out,file_hash(out/'result.json'),current)

    def test_independent_oracle_rejects_consistently_rehashed_wrong_reported_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,_,current=saved_fixture(Path(tmp));result=json.loads((out/'result.json').read_text());row=result['rows'][0]
            row['metrics']['fused']['oa']-=.1;metric_path=out/(row['name']+'-metrics.json');write_json(metric_path,row)
            result['artifacts'][metric_path.name]=file_hash(metric_path);write_json(out/'result.json',result)
            with self.assertRaisesRegex(ValueError,'Number differs'):verify_saved(out,file_hash(out/'result.json'),current)


if __name__=='__main__':unittest.main()
