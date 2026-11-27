"""Export only source, public class decisions and whitelisted aggregate receipts."""
import argparse
import json
from pathlib import Path
import zipfile

from c6_bridge import canonical,sha,frozen_source


def no_private_points(value):
    forbidden={'point_id','point_ids','samples','indices','box','boxes','parent_id','tile_id','tiles','batches','roles','local_centers'}
    if isinstance(value,dict):
        if set(value)&forbidden:raise ValueError('Private point/context field in review receipt')
        for child in value.values():no_private_points(child)
    elif isinstance(value,list):
        for child in value:no_private_points(child)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);root=Path(a.run_root);source=Path(__file__).parent
    def read(name):return json.loads((root/name).read_text())
    facts=read('review02/primary-custody01.json');envelope=read('plan06/common2-plan02.json');probe=read('validation07/common2-probe02.json');audit=read('validation07/common2-independent02.json');backup=read('preservation07.tar.gz.receipt.json');stale=read('validation07/stale-plan07.json');admission=json.loads((source/'admission.json').read_text())
    if not facts['passed'] or not probe['passed'] or not audit['passed'] or backup['mismatches'] or not backup['restored_all_phase_input_outputs_match_original'] or not stale['old_typed_key_plan_rejected']:raise ValueError('Completed metadata/input/audit/recovery evidence required')
    if probe['plan_sha256']!=admission['plan_sha256'] or audit['plan_sha256']!=admission['plan_sha256'] or sha(canonical(envelope['plan']))!=admission['plan_sha256']:raise ValueError('Final evidence plan differs')
    subset=[]
    for scene in envelope['plan']['scenes']:
        subset.append({key:scene[key] for key in ('scene','original_bank_pin','shape','counts','excluded_shared_parent_count','selected_parent_overlap','cross_role_observation_box_overlap_pixels','minimum_role_box_gap_pixels','normalization_statistics_sha256','processed_band_metadata_sha256')})
    safe_probe={key:probe[key] for key in ('producer','passed','plan_sha256','unique_original_points','point_presentations_two_directions','rejected_requests_before_arrays','torch_imported','new_model_calls','new_optimizer_updates','new_label_memberships','new_raw_GT_or_query_reads','independent_confirmation')}
    safe_probe['directions']=[{key:direction[key] for key in ('source','target','phases','payload_reads','payload_bytes')} for direction in probe['directions']]
    check=root/'check07/repository07.log';summary={'schema':'paper_c_C7_deidentified_review_v1','packager':a.expected_commit,'base':'ed57e389fd83ca9693a973f97168106ca8534074','issue':525,'classification_admission':'Metadata-corroborated two-class development input pair only','mapping_limit':'Derived MAT conversion to publisher bytes not certified; class mapping is explicitly corroborated inference. No accuracy, independent confirmation or causal-device claim.','first_observed_clock_tool_UTC':'2026-11-26T21:18:49Z','client_date':'2026-11-27','effective_work_seconds':None,'wait_seconds':None,'tests':{'source':'11f29ac2c7de9c528cfe6f7f3aab60b1d9a3f0ca','passed':11},'real_input_producer_and_auditor':'102530ff102c842714378527ebdc2aeb8173fee4','final_plan_sha256':admission['plan_sha256'],'private_probe_file_sha256':sha((root/'validation07/common2-probe02.json').read_bytes()),'repository_check':{'source':'102530ff102c842714378527ebdc2aeb8173fee4','exit_code':1,'reason':'Inherited stale current-status Markdown; shared ledger untouched','log_sha256':sha(check.read_bytes())},'CI_run_claim':False,'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'new_raw_GT_or_query_reads':0,'new_raw_image_or_GT_downloads':0,'payload_or_point_coordinate_files_on_Windows':0,'historical8337564720_byte_archive_complete':False}
    receipts={'review.json':summary,'primary-metadata-and-custody.json':facts,'selected-budget-and-context.json':subset,'input-probe-aggregates.json':safe_probe,'independent-audit.json':audit,'private-recovery.json':backup,'old-plan-rejection.json':stale}
    for value in receipts.values():no_private_points(value)
    catalog=root/'catalog07/all-original-class-decisions.csv'
    if sha(catalog.read_bytes())!='e072825bcec09f4c23a84039728d5471e99363d5d3d2dadfc6d075165080c49e':raise ValueError('All-original-class catalog differs')
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.py','.md','.json'):z.write(path,'source/'+path.name)
        for name,value in receipts.items():z.writestr('receipts/'+name,json.dumps(value,indent=2))
        z.write(catalog,'receipts/all-original-class-decisions.csv');z.write(root/'plan06/tests06.log','receipts/tests06.log')
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None or any(not name.startswith(('source/','receipts/')) for name in z.namelist()):raise ValueError('Review ZIP CRC/roster failure')
        count=len(z.namelist())
    receipt={'schema':'paper_c_C7_review_zip_receipt_v1','source':a.expected_commit,'bytes':out.stat().st_size,'sha256':sha(out.read_bytes()),'members':count,'raw_payload_or_point_coordinate_files':0};out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
