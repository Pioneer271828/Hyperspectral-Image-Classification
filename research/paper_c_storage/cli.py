"""Explicit optional-v2 importer/reader for existing synthetic C13 states."""
import argparse
import json
from pathlib import Path

from .benchmark import BASE,clean
from .bridge import import_case,materialize
from .codec import require
from .store import Store


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pack','restore','verify'])
    p.add_argument('--expected-commit',required=True);p.add_argument('--store',required=True);p.add_argument('--stream',required=True)
    p.add_argument('--source');p.add_argument('--out');p.add_argument('--layout',choices=['chunk-zlib','zip-xor'],default='zip-xor')
    p.add_argument('--max-bytes',type=int,default=64*1024**2);a=p.parse_args();clean(a.expected_commit)
    target=Path(a.store).resolve();require(target.is_relative_to(BASE/'runs') or target.is_relative_to(BASE/'backups'),'Authorized store root required')
    if a.command=='pack':
        source=Path(a.source).resolve();require(source.is_relative_to(BASE/'runs/paper-c-spatial-baseline-20261127-01'),'Only existing C13 generated fixtures admitted now')
        c=json.loads((source/'contract.json').read_text());require('synthetic' in c['scope'] and c['query_reads']==0,'Synthetic source contract required')
        store=Store(target,layout=a.layout,max_bytes=a.max_bytes);value=import_case(store,source,a.stream)
    else:
        store=Store(target);rows=store.frames(a.stream)
        if a.command=='restore':
            dest=Path(a.out).resolve();require(dest.is_relative_to(BASE/'runs') or dest.is_relative_to(BASE/'backups'),'Authorized restoration root required')
            value={'restored':str(materialize(store,a.stream,dest)),'stages':len(rows)}
        else:
            count=0
            for row in rows:
                for name in row['files']:store.decode_file(row,name);count+=1
            value={'original_files_verified':count,'stages':len(rows),'new_neural_calls':0,'new_optimizer_updates':0}
    store.checkpoint();store.close();print(json.dumps(value))


if __name__=='__main__':main()
