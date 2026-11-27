"""Use the frozen C7 gate under the separately registered C8 learning authority."""
import json
import os
from pathlib import Path
import subprocess
import sys

BASE='77081efa4862f6358dcce290a098e0f232fec5f8'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/paper_c_hsi_pair'))
from pair_inputs import PairInputs,phase_binding
from c6_bridge import canonical,sha,patch_batch,check_source as _c7_source


def protocol():return json.loads((Path(__file__).parent/'protocol.json').read_text())


def require_learning_enabled():
    """Fail before input access while the current user instruction pauses training."""
    if protocol().get('execution_state')!='real_learning_enabled':
        raise RuntimeError('Research training is paused; only synthetic engineering checks may run')


def clean(expected):
    _c7_source(expected)
    subprocess.run(['git','merge-base','--is-ancestor',BASE,expected],cwd=ROOT,check=True,capture_output=True)
    if subprocess.check_output(['git','diff',BASE,expected,'--','research/paper_c_hsi_pair'],cwd=ROOT):raise ValueError('C7 data admission changed')


def fresh(expected,path):
    clean(expected);path=Path(path).resolve()
    if os.name!='posix' or path.exists() or path.is_relative_to(ROOT):raise ValueError('New external Linux output required')
    return path


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):h.update(raw)
    return h.hexdigest()
