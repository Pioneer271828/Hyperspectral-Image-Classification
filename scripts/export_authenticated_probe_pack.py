"""Bind accepted historical single-case features to authorized K10/V5 probes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.authenticated_probe_pack import export
from research_runtime import code_snapshot


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit','capture','capture-sha256','freeze','analysis','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv);code=code_snapshot()
    if sys.platform!='linux' or not __debug__ or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Clean expected Linux consumer required')
    r=export(a.capture,a.capture_sha256,a.freeze,a.analysis,a.out,code)
    print(json.dumps({k:r[k] for k in ('status','case','train_rows','val_rows','model_forwards','probe_updates')}))


if __name__=='__main__':main()
