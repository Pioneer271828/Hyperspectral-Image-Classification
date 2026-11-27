"""Optional lossless import/materialization boundary for frozen C3 cases."""
import json
from pathlib import Path
import sys
from time import perf_counter

from .store import Store
from .codec import require,hashed,create,fsync_dir

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/paper_c_recovery'))
from transactions import inspect


def import_case(store,source,name):
    source=Path(source);contract=(source/'contract.json').read_bytes();c=json.loads(contract)
    rows,done=inspect(source,c);require(done,'Only complete immutable source cases admitted by importer')
    timings=[];identities=[]
    for step,row in enumerate(rows):
        folder=source/'stages'/f'{step:06d}'
        files={n:(folder/n).read_bytes() for n in ['state.pt','record.json','intent.json']}
        require(hashed(files['state.pt'])==row['state_sha256'],'Frozen source state differs')
        t=perf_counter()
        with store.transaction(name,step,files['intent.json'],contract) as tx:tx.commit(files)
        before_checkpoint=store.bytes();store.checkpoint()
        timings.append({'step':step,'encode_commit_checkpoint_seconds':perf_counter()-t,
                        'before_journal_checkpoint_bytes':before_checkpoint,'after_checkpoint_bytes':store.bytes()})
        identities.append({'step':step,'state_sha256':row['state_sha256'],'record_sha256':hashed(files['record.json'])})
    store.seal(name,(source/'done.json').read_bytes());store.checkpoint()
    # Original source state and metadata must still satisfy the frozen inspector.
    require(inspect(source,c)[1],'Source changed during import')
    return {'name':name,'source_contract_sha256':hashed(contract),'stages':len(rows),'identities':identities,'timings':timings}


def materialize(store,name,destination,*,through=None,forensic=False,fault=lambda phase:None):
    """New output only; arbitrary historical prefixes are forensic test replay.

    Pending intent always blocks ordinary restoration for continued training.
    Forensic=True does not authorize optimizer replay or erase the intent.
    """
    rows=store.frames(name)
    pending=store.db.execute('SELECT 1 FROM intents WHERE name=?',(name,)).fetchone()
    require(forensic or pending is None,'Uncertain intent prohibits training restoration')
    end=len(rows)-1 if through is None else through
    require(type(end) is int and 0<=end<len(rows),'Requested committed stage missing')
    require(forensic or end==len(rows)-1,'Only latest committed stage can be a normal continuation')
    dest=Path(destination);dest.mkdir(mode=0o700,exist_ok=False)
    # Original C3 inspect rejects this extra file before constructing a task.
    # It remains after any interrupted materialization, including an empty prefix.
    marker=dest/'.restore-pending';create(marker,b'Unpublished restoration; never resume training')
    (dest/'stages').mkdir();fsync_dir(dest)
    contract,seal=store.db.execute('SELECT contract,seal FROM streams WHERE name=?',(name,)).fetchone()
    create(dest/'contract.json',contract)
    for step,row in enumerate(rows[:end+1]):
        folder=dest/'stages'/f'{step:06d}';folder.mkdir()
        for key in ['state.pt','record.json','intent.json']:
            create(folder/key,store.decode_file(row,key));fault('after_file')
        fsync_dir(folder)
    fsync_dir(dest/'stages')
    if seal is not None and end==len(rows)-1:create(dest/'done.json',bytes.fromhex(json.loads(seal)['original_done_hex']))
    fsync_dir(dest);fault('before_ready')
    marker.unlink();fsync_dir(dest)
    # Inspect exact original state/intent/record SHA chains, not just tensors.
    inspect(dest,json.loads(contract))
    return dest
