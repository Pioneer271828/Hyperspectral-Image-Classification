"""Paired, scene-stratified spatial-cluster percentile diagnostics.

Conditional approximate intervals only: neither hashes nor disjoint identifiers
prove independent sampling. No policy selection or automatic deployment release.
"""
from copy import deepcopy
import numpy as np
from model.research_config import digest
from evaluation.dynamic_curve import require, pin, number


def statistics_from_counts(c):
    """Last two axes are true x predicted; missing classes remain undefined."""
    c = np.asarray(c, dtype=np.float64)
    support = c.sum(-1)
    true = np.diagonal(c, axis1=-2, axis2=-1)
    recall = np.divide(true, support, out=np.full_like(true, np.nan), where=support > 0)
    total = support.sum(-1)
    oa = np.divide(true.sum(-1), total, out=np.full_like(total, np.nan), where=total > 0)
    aa = recall.mean(-1)
    return np.concatenate((oa[..., None], aa[..., None], recall), axis=-1)


class PairedClusterProtocol:
    def __init__(self, spec, *, expected_hash):
        require(isinstance(spec, dict) and digest(spec) == expected_hash, 'External uncertainty protocol mismatch')
        require(set(spec) == {'schema', 'role', 'candidate_policy_hash', 'reference_state_hash', 'data_hash', 'units',
                             'classes', 'alpha', 'replicates', 'seed', 'min_clusters', 'min_class_clusters',
                             'min_tail_count', 'max_drop', 'independence_evidence_sha256', 'method'}, 'Exact uncertainty fields required')
        require(spec['schema'] == 'dy-paired-cluster-v1' and spec['role'] == 'development_confirmation', 'Only fixed development confirmation permitted')
        require(all(pin(spec[k]) for k in ('candidate_policy_hash','reference_state_hash','data_hash')), 'Fixed identities required')
        require(spec['independence_evidence_sha256'] is None or pin(spec['independence_evidence_sha256']), 'Invalid evidence reference')
        require(spec['method'] == 'paired_scene_cluster_percentile_bonferroni_lower', 'Registered approximate interval method required')
        require(type(spec['classes']) is int and spec['classes'] >= 2, 'Class count required')
        require(number(spec['alpha']) and 0 < spec['alpha'] < .5, 'Finite one-sided family alpha required')
        for key in ('min_clusters','min_class_clusters'):
            require(type(spec[key]) is int and spec[key] >= 2, 'At least two sampling clusters required')
        require(type(spec['min_tail_count']) is int and spec['min_tail_count'] >= 10, 'Explicit tail resolution floor >=10 required')
        require(type(spec['replicates']) is int and 20 <= spec['replicates'] <= 100000, 'Registered bounded replication budget required')
        require(type(spec['seed']) is int and 0 <= spec['seed'] < 2**32, 'Registered local RNG seed required')
        require(isinstance(spec['max_drop'], dict) and set(spec['max_drop']) == {'oa','aa','recall'}
                and all(number(v) and 0 <= v <= 1 for v in spec['max_drop'].values()), 'Explicit probability-scale margins required')
        units = spec['units'];require(isinstance(units, list) and units, 'Frozen cluster units required')
        ids = set(); identities = set(); groups = {}
        for row in units:
            require(isinstance(row, dict) and set(row) == {'id','scene','condition','cluster'}
                    and all(isinstance(v,str) and v for v in row.values()), 'Exact unit identity required')
            key = (row['scene'],row['condition'],row['cluster'])
            require(row['id'] not in ids and key not in identities, 'Repeated unit/cluster-condition pair')
            ids.add(row['id']);identities.add(key)
            groups.setdefault(row['scene'],{}).setdefault(row['condition'],set()).add(row['cluster'])
        require(all(len({tuple(sorted(v)) for v in group.values()}) == 1 for group in groups.values()), 'Every condition must contain the same scene clusters')
        draws = sum(len(next(iter(v.values()))) for v in groups.values()) * spec['replicates']
        require(draws <= 5000000, 'Explicit resampling allocation exceeds bounded engineering limit')
        self._spec = deepcopy(spec);self.protocol_hash = expected_hash

    @property
    def specification(self):return deepcopy(self._spec)

    def report(self, records, *, expected_records_hash):
        s = self._spec;require(digest(s) == self.protocol_hash, 'Protocol changed')
        require(pin(expected_records_hash) and digest(records) == expected_records_hash == s['data_hash'], 'Frozen paired data mismatch')
        require(isinstance(records, list) and len(records) == len(s['units']), 'Complete paired units required')
        units = {u['id']:u for u in s['units']};seen = set();groups = {};c = s['classes']
        for row in records:
            require(isinstance(row,dict) and set(row) == {'id','candidate_policy_hash','reference_state_hash','candidate','reference'}, 'Exact paired evidence required')
            require(row['id'] in units and row['id'] not in seen, 'Unknown/repeated paired unit')
            require(row['candidate_policy_hash'] == s['candidate_policy_hash'] and row['reference_state_hash'] == s['reference_state_hash'], 'Frozen models changed')
            seen.add(row['id']);u = units[row['id']];arrays = []
            for field in ('candidate','reference'):
                raw = row[field]
                require(isinstance(raw,list) and len(raw) == c and all(isinstance(v,list) and len(v) == c for v in raw), 'Exact confusion shape required')
                require(all(type(x) is int and 0 <= x <= 2**31-1 for v in raw for x in v), 'Exact bounded integer counts required')
                arrays.append(np.array(raw,dtype=np.float64))
            require(np.array_equal(arrays[0].sum(1),arrays[1].sum(1)) and arrays[0].sum() > 0, 'Both models must score identical label support')
            groups.setdefault(u['scene'],{}).setdefault(u['condition'],{})[u['cluster']] = np.stack(arrays)
        require(sum(v.sum() for scene in groups.values() for condition in scene.values() for v in condition.values()) < 2**40, 'Counts exceed exact arithmetic envelope')
        rng = np.random.default_rng(s['seed']);B=s['replicates'];descriptions=[];observed=[];boot=[];support_counts=[];draw_hashes={}
        pooled = np.zeros((2,c,c));pooled_boot=np.zeros((B,2,c,c));all_counts=[];all_cluster_numbers=[]
        for scene,conditions in sorted(groups.items()):
            clusters=sorted(next(iter(conditions.values())));n=len(clusters);all_cluster_numbers.append(n)
            # The same draws apply to both models and every condition in a scene.
            indices=rng.integers(0,n,size=(B,n));weights=np.zeros((B,n),dtype=np.int64)
            np.add.at(weights,(np.arange(B)[:,None],indices),1)
            draw_hashes[scene]=dict(clusters=clusters,indices_hash=digest(indices.tolist()))
            class_present=np.zeros((n,c),dtype=bool)
            for condition,by_cluster in sorted(conditions.items()):
                data=np.stack([by_cluster[k] for k in clusters]);counts=data.sum(0)
                samples=np.einsum('bn,nkij->bkij',weights,data)
                class_clusters=(data[:,0].sum(-1)>0).sum(0)
                class_present |= data[:,0].sum(-1)>0
                descriptions.append(dict(scene=scene,condition=condition,clusters=n,class_clusters=class_clusters.tolist()))
                support_counts.append(class_clusters);observed.append(counts);boot.append(samples)
                pooled+=counts;pooled_boot+=samples
            all_counts.append(class_present.sum(0))
        descriptions.insert(0,dict(scene='__pooled__',condition='__all__',clusters=sum(all_cluster_numbers),class_clusters=np.sum(all_counts,axis=0).tolist()))
        observed.insert(0,pooled);boot.insert(0,pooled_boot);support_counts.insert(0,np.sum(all_counts,axis=0))
        dimensions=len(observed)*(c+2);tail=s['alpha']/dimensions
        tail_adequate=int(np.floor(B*tail))>=s['min_tail_count'];entries=[]
        metric_names=['oa','aa']+['recall_'+str(i+1) for i in range(c)]
        for group,counts,samples,cs in zip(descriptions,observed,boot,support_counts):
            point=statistics_from_counts(counts);diff=point[0]-point[1]
            replicates=statistics_from_counts(samples);d=replicates[:,0]-replicates[:,1]
            for j,name in enumerate(metric_names):
                values=d[:,j];finite=np.isfinite(values);issues=[]
                if not np.isfinite(diff[j]):issues.append('missing_observed_class')
                if not finite.all():issues.append('missing_class_in_resample')
                if group['clusters']<s['min_clusters'] or (group['scene']=='__pooled__' and min(all_cluster_numbers)<s['min_clusters']):issues.append('too_few_scene_clusters')
                needed=int(min(cs)) if j<2 else int(cs[j-2])
                if needed<s['min_class_clusters']:issues.append('too_few_class_clusters')
                if not tail_adequate:issues.append('insufficient_tail_resolution')
                if s['independence_evidence_sha256'] is None:issues.append('independence_evidence_missing')
                degenerate=bool(finite.all() and np.ptp(values)==0)
                if degenerate:issues.append('degenerate_resampling_distribution')
                lower=float(np.quantile(values,tail,method='lower')) if finite.all() else None
                margin=s['max_drop'][name if j<2 else 'recall']
                clears=not issues and lower > -margin
                entries.append(dict(group=group,metric=name,candidate=float(point[0,j]) if np.isfinite(point[0,j]) else None,
                    reference=float(point[1,j]) if np.isfinite(point[1,j]) else None,difference=float(diff[j]) if np.isfinite(diff[j]) else None,
                    lower_percentile=lower,max_drop=margin,finite_replicates=int(finite.sum()),
                    sample_sd=float(values.std(ddof=1)) if finite.all() else None,issues=issues,
                    approximate_bound_clears_margin=bool(clears),replicates=[float(x) if np.isfinite(x) else None for x in values]))
        passed=all(e['approximate_bound_clears_margin'] for e in entries)
        return dict(format='dy-paired-cluster-report-v1',protocol_hash=self.protocol_hash,records_hash=expected_records_hash,
            entries=entries,draws=draw_hashes,simultaneous_dimensions=dimensions,one_sided_marginal_tail=tail,
            quantile_rule='lower_order_statistic',tail_expected_count=B*tail,
            status='approximate_bounds_clear_margins_conditional_on_sampling_assumptions' if passed else 'inconclusive_or_not_protected',
            inferential_scope='One fixed model pair; scene-stratified cluster resampling, pooled pixel-weighted metrics. Not training-seed uncertainty.',
            method_limit='Percentile approximation, not BCa or a finite-sample coverage guarantee; independence evidence must be reviewed externally.',
            automatic_deployment_authorized=False,statistical_noninferiority_verified=False,policy_selection=False,optimizer_updates=0,query=0)
