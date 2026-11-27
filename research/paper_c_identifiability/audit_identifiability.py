"""Independent factorial and bounded exhaustive audit of exported metadata."""
import argparse
from collections import defaultdict,Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    raw=Path(a.metadata).read_bytes();value=json.loads(raw);rows=[]
    for case in value['cases']:
        rs=case['requests'];results={}
        for family,record in case['nulls'].items():
            groups=defaultdict(list)
            for r in rs:
                mask=r['mask_indices']
                expected=hashlib.sha256(json.dumps(mask,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if len(mask)!=r['K'] or sorted(set(mask))!=mask or expected!=r['mask_sha256']:raise ValueError('Column identity differs')
                if family=='legacy':key=(r['view'],tuple(r['shape'] or []))
                elif family=='shape_only':key=tuple(r['shape'] or [])
                else:key=(r['context'],len(mask),min(mask),max(mask))
                groups[key].append(r)
            total=1;quotient=1
            for g in groups.values():
                n=len(g);k=sum(r['full_selected']['content_coverage'] for r in g)
                total*=math.factorial(n)//math.factorial(k)//math.factorial(n-k)
                if family=='same_context_count_span':
                    aliases=Counter((r['context'],tuple(r['mask_indices'])) for r in g)
                    if len(g)>4:raise ValueError('Exhaustive same-context audit bound exceeded')
                    quotient*=sum(sum(v)==k for v in itertools.product(*(range(m+1) for m in aliases.values())))
            if str(total)!=record['assignments']:raise ValueError('Independent allocation support differs')
            if family=='same_context_count_span' and str(quotient)!=record['alias_quotient_assignments']:raise ValueError('Independent input-quotient support differs')
            results[family]=str(total)
        if case['admission']['admitted'] or case['native_retained_column_map_verified']:raise ValueError('Unexpected real-device admission requires new evidence review')
        rows.append({'case':case['case'],'exact_support':results,'internal_alias_quotient':case['nulls']['same_context_count_span']['alias_quotient_assignments']})
    result={'status':'PASSED_INDEPENDENT_METADATA_AUDIT','metadata_sha256':hashlib.sha256(raw).hexdigest(),'cases':rows,
        'new_model_or_observation_array_reads':0,'new_labels_or_prediction_reads':0,'new_neural_calls':0,'new_optimizer_updates':0}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(rows),'new_neural_calls':0}))


if __name__=='__main__':main()
