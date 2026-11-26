"""Independent saved-probability recount and WP12-facing position supplement; no model forwards."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts.run_wp07_condition_integration import table
from transfer import registered_development as dev
from transfer.wp07_graph_range import require
from transfer.xuzhou_multiscale_audit import _exact
from transfer import wp07_exact_positions as positions
from transfer import wp07_observed_conditions as summary_encoder
from transfer.native_scale_training import baseline


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(points, metrics, classes):
    groups = defaultdict(list)
    for row in points:
        groups[row['head'], row['band']].append(row)
    maximum = 0.
    for (head, band), rows in groups.items():
        cm = [[0]*classes for _ in range(classes)]
        for row in rows:
            cm[row['class_id']-1][max(range(classes), key=row['probabilities'].__getitem__)] += 1
        nll = math.fsum(-math.log(max(row['probabilities'][row['class_id']-1], 1e-8)) for row in rows)/len(rows)
        reference = metrics[head][band]
        require(cm == reference['confusion_matrix'], 'Independent confusion matrix mismatch')
        error = abs(nll-reference['nll'])
        require(error <= 1e-12, 'Independent NLL mismatch')
        maximum = max(maximum, error)
    return len(groups), maximum


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def alias_proof(left,right,classes,seed,*,strip_rotation=False):
    comparable={k:v for k,v in right['model'].items() if not (strip_rotation and k.endswith('.position_rotation'))}
    require(_exact(left['model'],comparable) and _exact(left['rng'],right['rng']),'Condition-negative predictor or RNG changed')
    model=positions.initialize(baseline(classes,seed),seed+7321)
    names=list(dict(model.named_parameters()))
    require(len(names)==len(left['optimizer']['param_groups'][0]['params'])==len(right['optimizer']['param_groups'][0]['params']),
            'Optimizer parameter inventory differs from the pinned canonical architecture')
    excluded={i for i,name in enumerate(names) if '.indices.' in name or '.exact_mask.' in name}
    def shared(optimizer):
        return {**optimizer,'state':{i:v for i,v in optimizer['state'].items() if i not in excluded}}
    require(_exact(shared(left['optimizer']),shared(right['optimizer'])),'Shared/active optimizer state changed')
    for state in (left,right):
        for i in excluded & state['optimizer']['state'].keys():
            value=state['optimizer']['state'][i]
            require(not bool(torch.count_nonzero(value['exp_avg'])) and not bool(torch.count_nonzero(value['exp_avg_sq'])),
                    'Masked condition encoder has nonzero Adam moments')
    return dict(predictor_and_RNG_exact=True,shared_active_optimizer_exact=True,
                full_optimizer_exact=_exact(left['optimizer'],right['optimizer']),
                inactive_projection_moments_zero=True,
                fixed_rotation_buffers_excluded=strip_rotation)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    a=p.parse_args();code=code_snapshot();require_frozen_code(code)
    require(code['commit']==a.expected_commit,'Wrong evidence-consumer source')
    a.out.mkdir()
    qualification=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.random.default_generator.manual_seed(27183)
        encoder=positions.PositionEncoder(summary_encoder.IndexConditionEncoder())
        for scene,n in (('paviaU',103),('xuzhou',436)):
            proposed=positions.proposed_real_positions(n)
            for representation in positions.SUPPORTED_REPRESENTATIONS:
                encoder.representation=representation;encoded={}
                for view,indices in proposed.items():
                    value=encoder(summary_encoder.IndexPacket(None,indices,n),None,None,encoder.exact_mask[0].weight)
                    encoded[view]=value
                internal_equal=torch.equal(encoded['shared_a'],encoded['interior_b'])
                require(internal_equal==(representation in ('summary8','balanced_summary')),'Wrong metadata-only internal identifiability')
                require(not torch.equal(encoded['shared_a'],encoded['shift_b']),'Shift descriptor not identifiable')
                qualification.append(dict(scene=scene,representation=representation,internal_encoding_exact=internal_equal,
                    shift_encoding_distinct=True,classifier_forwards=0,new_real_views_materialized=0,
                    source_indices=digest(proposed),basis='fixed positions, no new-view scores or model selection'))
    table(a.out/'all-representation-metadata-qualification.csv',qualification)
    results=[];all_endpoints=[];all_classes=[];all_params=[];all_cost=[];all_controls=[]
    matrices=0;max_error=0.;aliases=[];pins=[]
    for family,folder in (('legacy','matrix-v1-output'),('balanced','balanced-v2-output')):
        source=a.root/folder;summary=load(source/'summary.json');require(summary['status']=='passed','Matrix incomplete')
        pins.append(dict(family=family,path=str(source/'summary.json'),sha256=sha(source/'summary.json'),
                         producer=summary['producer'],source_hash=summary['source_hash']))
        require(len(summary['cases'])==48,'Unexpected matrix case roster')
        for case in summary['cases']:
            dest=source/case['case'];audit=load(dest/'independent.json')
            for endpoint in audit['endpoints'].values():
                count,error=recount(endpoint['original_point_predictions'],endpoint['metrics'],case['classes'])
                matrices+=count;max_error=max(max_error,error)
            results.append({**case,'family':family})
        for file,bucket in (('endpoints.csv',all_endpoints),('per-class.csv',all_classes),
                            ('parameter-participation.csv',all_params),('resident-cost.csv',all_cost),('controls.csv',all_controls)):
            bucket.extend(dict(family=family,**row) for row in read_csv(source/file))
        reps=['summary8','exact_mask'] if family=='legacy' else ['balanced_summary','balanced_exact']
        for classes in (9,22):
            for seed in (64,512,2048):
                for mode in ('mean','content'):
                    states=[torch.load(source/f'c{classes}-s{seed}-{rep}-both.{mode}'/'full.pt',map_location='cpu',weights_only=False) for rep in reps]
                    proof=alias_proof(states[0],states[1],classes,seed)
                    aliases.append(dict(family=family,classes=classes,seed=seed,mode=mode,**proof,
                                        separate_interface_cases_are_not_independent_scientific_arms=True))
    real=load(a.root/'real-probe-v1-output/_private/point-records.json')
    for classes in (9,22):
        for seed in (64,512,2048):
            for mode in ('mean','content'):
                legacy=torch.load(a.root/'matrix-v1-output'/f'c{classes}-s{seed}-summary8-both.{mode}'/'full.pt',map_location='cpu',weights_only=False)
                balanced=torch.load(a.root/'balanced-v2-output'/f'c{classes}-s{seed}-balanced_summary-both.{mode}'/'full.pt',map_location='cpu',weights_only=False)
                proof=alias_proof(legacy,balanced,classes,seed,strip_rotation=True)
                aliases.append(dict(family='cross-family',classes=classes,seed=seed,mode=mode,**proof,
                                    separate_interface_cases_are_not_independent_scientific_arms=True))
    real_matrices=0
    for record in real['records']:
        count,error=recount(record['points'],record['metrics'],9)
        real_matrices+=count;max_error=max(max_error,error)
    table(a.out/'synthetic-endpoints.csv',all_endpoints)
    table(a.out/'synthetic-classes.csv',all_classes)
    table(a.out/'parameter-participation.csv',all_params)
    table(a.out/'synthetic-resident-cost.csv',all_cost)
    table(a.out/'same-content-controls.csv',all_controls)
    table(a.out/'mean-content-aliases.csv',aliases)
    for relative in ('real-probe-v1-output/endpoints.csv','real-cost-v1-output/real-resident-cost.csv',
                     'admission-v3-output/existing-provider-reads.csv','admission-v3-output/pre-score-identifiability.csv'):
        shutil.copyfile(a.root/relative,a.out/Path(relative).name)
    index={tuple(row[key] for key in ('family','classes','seed','representation','policy','endpoint','head','band')):row for row in all_endpoints}
    class_index=defaultdict(list)
    for row in all_classes:
        class_index[tuple(row[key] for key in ('family','classes','seed','representation','policy','endpoint','head','band'))].append(row)
    pairs=[]
    for key,right in index.items():
        family,classes,seed,representation,policy,endpoint,head,band=key
        comparison=[]
        if policy=='both.joint':
            comparison.append(('joint-content',index[family,classes,seed,representation,'both.content',endpoint,head,band]))
        if representation in ('exact_mask','balanced_exact'):
            baseline='summary8' if family=='legacy' else 'balanced_summary'
            comparison.append(('exact-summary',index[family,classes,seed,baseline,policy,endpoint,head,band]))
        for contrast,left in comparison:
            left_key=tuple(left[k] for k in ('family','classes','seed','representation','policy','endpoint','head','band'))
            left_classes={r['class_id']:r for r in class_index[left_key]}
            minimum=min(float(r['recall'])-float(left_classes[r['class_id']]['recall']) for r in class_index[key])
            pairs.append(dict(family=family,classes=classes,seed=seed,representation=representation,policy=policy,
                endpoint=endpoint,head=head,band=band,contrast=contrast,nll_delta=float(right['nll'])-float(left['nll']),
                oa_delta=float(right['oa'])-float(left['oa']),minimum_class_recall_delta=minimum,protected=minimum>=-.02))
    table(a.out/'paired-comparisons.csv',pairs)
    base=a.root.parent/'wp07-b-fine16-20261126/wp12';prior=a.out/'unchanged-base';prior.mkdir()
    preserved={}
    for name in ('adverse-classes.csv','graph-response.csv','primary-pairs.csv','resident-cost.csv','xuzhou-classes-5-6.csv'):
        shutil.copyfile(base/name,prior/name);require(sha(base/name)==sha(prior/name),'Old WP12 table changed');preserved[name]=sha(prior/name)
    authority=load(a.root/'admission-v3-output/source-role-authority.json')
    xuzhou=next(r for r in authority if r['scene']=='xuzhou')
    require(xuzhou['new_position_views_admitted'] is False,'Never adopt a new view through evidence reporting')
    draft=dict(schema='wp07_B_controlled_positions_real_draft_v2',status='not_admitted_not_registered_not_executed',
        scene='xuzhou',bank=xuzhou['bank'],preflight=xuzhou['preflight'],
        positions=xuzhou['new_position_proposal'],position_hashes={k:digest(v) for k,v in xuzhou['new_position_proposal'].items()},
        questions={'overall_shift':['shared_a','shift_b'],'fixed_count_endpoints_span_internal_columns':['shared_a','interior_b']},
        retained_each=218,master_processed_count=436,shared_baseline_is_trained_once=True,
        policies=['content','joint_balanced_summary','joint_balanced_exact'],model_seeds=[64,512,2048],models=9,
        total_real_optimizer_updates_requested_as_draft_only=837,steps_per_model=93,prefix=46,resume=47,completed=0,
        training_exposure='31 released labelled blocks once under each of3 views; one shared_a exposure, not duplicated per question',
        validation_events=[0,46,93],selection='first minimum fused released V5 NLL; 3views of same45points',
        roles={'devtrain':45,'devval':45,'per_class_each_role':5,'support_seed':512,'validation_seed':3031},
        primary='joint_balanced_exact-content lowers fused NLL in all3 seeds and all3 views, every head/class recall delta >=-0.02',
        representation_control='joint_balanced_exact-joint_balanced_summary, common projection/normalization/initial state/candidates/budget; input rank and Adam geometry remain different',
        label_precision='One class error is1/5=20 percentage points. A2point recall floor allows no class loss.',
        model_and_condition_chosen_by='pre-score identifiability and capacity controls; no validation-score selection',
        physical_or_independent_spatial_confirmation=False,new_real_training_authorized=False,odd_view_authorized=False,
        ready_for_training=False,estimated_GPU3_end_to_end_minutes=[15,40],contention_additional=True,
        estimate_scope='planning range based on original62step development runs,3view vs2view exposure, measured resident readouts; includes registration/validation/independent audit/IO, not a measured train-time speed claim',
        release_requirements=['Explicit approval of these3 hash-pinned controlled views; ordinary Bank.sample capability is not admission',
            'Fresh bound-array and preprocessing/statistics/point/geometry preflight without rawGT/query',
            'Nondegenerate own-view Q/A and causal descriptor isolation, matching source/initial state/capacity',
            'Separate exact837update optimizer registration and native real-provider recovery qualification',
            'Physical-device arithmetic/lease and incremental backup custody'])
    dev._json_new(a.out/'next-real-matrix-draft.json',draft)
    receipt=dict(schema='wp12_B_exact_position_supplement_v1',status='engineering_and_read_only_development_diagnostics',
        executor=code['commit'],source_hash=code['source_hash'],inputs=pins,synthetic_cases=96,main_synthetic_updates=4608,
        synthetic_confusions_independently_recounted=matrices,real_confusions_independently_recounted=real_matrices,
        NLL_max_absolute_error=max_error,predictor_shared_optimizer_RNG_alias_pairs=len(aliases),
        full_optimizer_exact_alias_pairs=sum(row['full_optimizer_exact'] for row in aliases),
        distinct_policy_pairs_per_class_seed_per_family=6,mean_content_aliases_not_independent_arms=True,
        distinct_policy_variants_per_class_seed_across_families=10,
        real_optimizer_updates=0,new_real_views_materialized=0,condition_real_heads_trained=False,
        central_WP12_registry_modified=False,old_B_WP12_CSVs_byte_identical=preserved,
        next_real_ready=False,new_real_view_permission_not_inferred_from_API=True,
        classifier_forwards=0,optimizer_steps=0,query_accessed=False,raw_GT_opened=False)
    require(matrices==2304 and real_matrices==60 and len(aliases)==36,'Wrong evidence denominators')
    dev._json_new(a.out/'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
