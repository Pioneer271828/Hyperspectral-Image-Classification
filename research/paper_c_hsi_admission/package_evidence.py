"""Export source and explicitly whitelisted aggregates; no private point/array files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from inspect_bank import guard


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--authoritative-UTC',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=guard(a.expected_commit,a.out);root=Path(a.run_root);source=Path(__file__).resolve().parent
    audit=json.loads((root/'runs/validation07/independent-inputs01.json').read_text());probe=json.loads((root/'runs/validation11/development-probe01.json').read_text());spatial=json.loads((root/'runs/spatial09/strict-original-roles01.json').read_text());backup=json.loads((root/'preservation01.tar.gz.receipt.json').read_text())
    if not audit['passed'] or not probe['passed'] or backup['mismatches'] or backup['restored_verified_files']!=backup['source_files']:raise ValueError('Passed independent intake/probe/restore receipts required')
    subset=[]
    for scene in spatial['scenes']:
        subset.append({'scene':scene['scene'],'complete_K5_V5_original_numeric_classes':scene['common_K5_V5_feasible_original_classes'],'excluded_shared_parent_count':len(scene['excluded_shared_parents']),'roles':{role:{key:value for key,value in group.items() if key!='tiles'} for role,group in scene['roles'].items()},'new_array_reads':scene['new_array_reads'],'independent_confirmation':False})
    failed=[]
    for name,meaning in [('runs/inspection03/xuzhou-metadata01.log','Initial reader required unlabeled tile JSON absent from A labeled-only preservation; corrected without expanding admitted observations.'),('runs/check12/repository12.log','Repository check exit1: inherited stale current-status Markdown; shared ledger untouched.')]:
        path=root/name
        if path.exists():failed.append({'stage':name,'bytes':path.stat().st_size,'sha256':sha(path),'meaning':meaning,'full_log_retained_on_authorized_Linux':True})
    summary={'schema':'paper_c_C6_redacted_review_v1','packager_source':a.expected_commit,'authoritative_package_UTC':a.authoritative_UTC,'first_authoritative_UTC':'2026-11-26T19:49:28Z','effective_work_seconds':None,'wait_seconds':None,'module_tests':{'source':'3ce951a920a561cf234f36c9d71310f5b7e58a6d','passed':11},'independent_all_112_block_audit_source':audit['source'],'four_block_zero_model_probe_source':probe['source'],'spatial_subset_source':spatial['producer'],'private_restore':backup,'failed_or_incomplete_checks':failed,'model_accuracy_or_transfer_efficacy':'NOT_TESTED','common_semantic_head':'BLOCKED exact product/ordinal/data-rights materials','independent_confirmation':'NOT_RELEASED','new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'raw_GT_or_query_reads':0,'payload_or_point_coordinate_files_on_Windows':0,'CI_run_claim':False,'old_large_archive_gap':'Historical8337564720-byte archive still incomplete; this C6 snapshot does not close it.'}
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.py','.md','.json'):z.write(path,'source/'+path.name)
        for name,value in [('review.json',summary),('independent-inputs.json',audit),('development-probe.json',probe),('strict-parent-subset-aggregates.json',subset)]:z.writestr('receipts/'+name,json.dumps(value,indent=2))
        for name in ('runs/validation11/tests11.log','references01/tableIX-review01.json'):z.write(root/name,'receipts/'+Path(name).name)
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP CRC failure')
        names=z.namelist()
        if any(not n.startswith(('source/','receipts/')) or n.endswith(('.npy','.mat','.pdf','.tif','.png')) for n in names):raise ValueError('Unexpected review member')
    receipt={'schema':'paper_c_C6_review_zip_receipt_v1','source':a.expected_commit,'bytes':out.stat().st_size,'sha256':sha(out),'members':len(names),'payload_or_point_coordinate_files':0};out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
