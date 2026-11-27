"""Export only project source and aggregate receipts; keep payloads/coordinates off Windows."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from catalog import guard


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run',required=True);p.add_argument('--aerial-audit',required=True);p.add_argument('--paired-audit',required=True);p.add_argument('--reader-receipt',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out)
    if out.exists():raise ValueError('New export package required')
    run=Path(a.run);meta=json.loads((run/'ovh-official-metadata02/manifest.json').read_text());catalog=json.loads((run/'ovh-training-catalog01/manifest.json').read_text());intake=json.loads((run/'public-train-intake01/manifest.json').read_text());paired=json.loads((run/'paired-train-intake01/manifest.json').read_text())
    failures=[]
    for name,meaning in [('fetch-metadata01.log','LXC cannot reach HF API'),('ovh-fetch01.log','Initial background launch ended without data; foreground retry succeeded'),('audit01.log','Incorrect TiffPage GeoTIFF property access; preserved failure'),('audit02.log','Whole-set strict center gate rejected one real sample; split admission/quarantine implemented without changing tolerance')]:
        file=run/name
        if file.exists():failures.append({'log':name,'bytes':file.stat().st_size,'sha256':sha(file),'meaning':meaning,'raw_log_kept_on_authorized_Linux':True})
    provenance={'schema':'paper_c_redacted_intake_provenance_v1','packager':a.expected_commit,'metadata_producer':meta['producer'],'metadata_archives':meta['archives'],'metadata_json_receipts':meta['members'],'catalogs':catalog,'intake_producer':intake['producer'],'paired_producer':paired['producer'],'payload_receipts':[{'archive':r['archive'],'bytes':r['bytes'],'sha256':r['sha256'],'crc32':r['crc32']} for r in intake['members']+paired['members']],'payload_or_coordinate_files_exported':0,'new_model_calls':0,'new_optimizer_updates':0,'test_or_query_payload_reads':0,'failed_attempts':failures,'historical_backup_gap':{'archive_bytes':8337564720,'verified_prefix_bytes':535822336,'whole_archive_sha256':'b9cf9adc398d79214c23abd034e4178ff4e82c2fdd0fb2ba81fd5ac092fdb045','complete':False}}
    # ZIP-directory examples contain only train/validation filenames, but remove
    # these identifiers from the review receipt as they are unnecessary to QA.
    for row in provenance['catalogs']['archives'].values():row.pop('example_names',None)
    source=Path(__file__).resolve().parent
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.py','.md','.json'):z.write(path,'source/'+path.name)
        z.writestr('receipts/provenance.json',json.dumps(provenance,indent=2))
        for file,name in [(Path(a.aerial_audit),'aerial-audit.json'),(Path(a.paired_audit),'paired-audit.json'),(Path(a.reader_receipt),'reader-smoke.json')]:
            value=json.loads(file.read_text());z.writestr('receipts/'+name,json.dumps(value,indent=2))
        for name in ('tests05.log','tests08.log','tests09.log'):
            file=run/name
            if file.exists():z.write(file,'receipts/'+name)
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('Export ZIP CRC failure')
    print(json.dumps({'status':'REDACTED_REVIEW_PACKAGE','bytes':out.stat().st_size,'sha256':sha(out),'payload_or_coordinate_files':0}))


if __name__=='__main__':main()
