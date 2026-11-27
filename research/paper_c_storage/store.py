"""Optional v2 CAS using stdlib SQLite WAL/FULL, without a historical fork."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import sqlite3
import zlib

from .codec import CHUNK,MAX_FILE,MAX_DEPTH,require,hashed,encoded,segments,xor


class Store:
    def __init__(self,root,*,layout=None,max_bytes=512*1024**2):
        self.root=Path(root);self.max_bytes=max_bytes;self.read_bytes=0;self.write_bytes=0;self._heads={}
        new=not self.root.exists()
        if new:
            require(layout in ['chunk-zlib','zip-xor'],'Explicit new-store layout required');self.root.mkdir(mode=0o700)
        self.db=sqlite3.connect(self.root/'objects.sqlite3',timeout=0,isolation_level=None)
        self.db.execute('PRAGMA foreign_keys=ON');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA busy_timeout=0')
        require(self.db.execute('PRAGMA journal_mode=WAL').fetchone()[0]=='wal','WAL unavailable')
        if new:
            self.db.executescript('''
                CREATE TABLE settings(value BLOB NOT NULL);
                CREATE TABLE streams(name TEXT PRIMARY KEY,contract BLOB NOT NULL,seal BLOB);
                CREATE TABLE intents(name TEXT PRIMARY KEY REFERENCES streams(name),step INTEGER NOT NULL,raw BLOB NOT NULL);
                CREATE TABLE objects(sha TEXT PRIMARY KEY,n INTEGER NOT NULL,codec TEXT NOT NULL,base TEXT REFERENCES objects(sha),depth INTEGER NOT NULL,payload BLOB NOT NULL,payload_sha TEXT NOT NULL);
                CREATE TABLE frames(name TEXT REFERENCES streams(name),step INTEGER NOT NULL,manifest BLOB NOT NULL,sha TEXT NOT NULL,intent BLOB NOT NULL,PRIMARY KEY(name,step));
            ''')
            self.db.execute('INSERT INTO settings VALUES (?)',(encoded({'schema':'c14_sqlite_cas_v2','layout':layout,'chunk':CHUNK,'max_depth':MAX_DEPTH,'max_file':MAX_FILE}),))
        value=json.loads(self.db.execute('SELECT value FROM settings').fetchone()[0])
        require(value=={'schema':'c14_sqlite_cas_v2','layout':value['layout'],'chunk':CHUNK,'max_depth':MAX_DEPTH,'max_file':MAX_FILE}
                and value['layout'] in ['chunk-zlib','zip-xor'],'Store format differs')
        require(layout is None or layout==value['layout'],'Layout changed');self.layout=value['layout']

    def close(self):self.db.close()
    def bytes(self):return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
    def allocated(self):return sum(p.stat().st_blocks*512 for p in self.root.iterdir())
    def checkpoint(self):
        require(self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]==0,'WAL checkpoint busy')
        return self.bytes()

    @contextmanager
    def locked(self):
        with (self.root/'writer.lock').open('a+b') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Another writer owns this store') from None
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

    def get(self,key,*,depth=0,seen=None):
        require(depth<=MAX_DEPTH,'Delta chain exceeds bound')
        seen=set() if seen is None else seen;require(key not in seen,'Delta cycle');seen=seen|{key}
        row=self.db.execute('SELECT n,codec,base,depth,payload,payload_sha FROM objects WHERE sha=?',(key,)).fetchone()
        require(row is not None,'Missing content object');n,codec,base,level,payload,psha=row;self.read_bytes+=len(payload)
        require(0<n<=CHUNK and codec in ['raw','zlib','xor-zlib'] and 0<=level<=MAX_DEPTH and hashed(payload)==psha,'Object metadata/payload checksum differs')
        if codec=='raw':raw=payload
        else:
            d=zlib.decompressobj();raw=d.decompress(payload,n+1)
            require(d.eof and not d.unconsumed_tail and not d.unused_data,'Truncated or oversized compressed object')
        if codec=='xor-zlib':
            require(base is not None and level>0,'Missing delta base');old,bh=self.get(base,depth=depth+1,seen=seen)
            require(level==bh['depth']+1,'Delta depth differs');raw=xor(raw,old)
        else:require(base is None and level==0,'Unexpected object dependency')
        require(len(raw)==n and hashed(raw)==key,'Decoded object checksum differs')
        return raw,{'depth':level,'codec':codec,'bytes':n,'payload_bytes':len(payload)}

    def put(self,raw,base=None):
        key=hashed(raw)
        if self.db.execute('SELECT 1 FROM objects WHERE sha=?',(key,)).fetchone():
            require(self.get(key)[0]==raw,'Duplicate object differs');return key
        choices=[('raw',raw,None,0),('zlib',zlib.compress(raw,1),None,0)]
        if self.layout=='zip-xor' and base is not None:
            old,bh=self.get(base)
            if len(old)==len(raw) and bh['depth']<MAX_DEPTH:
                choices.append(('xor-zlib',zlib.compress(xor(raw,old),1),base,bh['depth']+1))
        codec,payload,parent,level=min(choices,key=lambda v:len(v[1])+(64 if v[2] else 0))
        self.db.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?)',(key,len(raw),codec,parent,level,payload,hashed(payload)))
        self.write_bytes+=len(payload);return key

    def frames(self,name,*,writable=False):
        stream=self.db.execute('SELECT contract,seal FROM streams WHERE name=?',(name,)).fetchone()
        if stream is None:return []
        contract,seal=stream
        if writable:
            require(self.db.execute('SELECT 1 FROM intents WHERE name=?',(name,)).fetchone() is None,'Uncertain pending intent blocks automatic continuation')
            require(seal is None,'Sealed stream cannot append')
        rows=[];previous=None
        for step,raw,digest,intent in self.db.execute('SELECT step,manifest,sha,intent FROM frames WHERE name=? ORDER BY step',(name,)):
            require(step==len(rows) and hashed(raw)==digest,'Frame gap/checksum differs');m=json.loads(raw)
            require(m['schema']=='c14_frame_v2' and m['step']==step and m['previous']==previous and m['contract_sha256']==hashed(contract)
                    and m['intent_sha256']==hashed(intent),'Frame contract/intent chain differs')
            previous=digest;rows.append(m)
        if seal is not None:
            s=json.loads(seal);require(s['frames']==len(rows) and s['last_frame']==previous and s['contract_sha256']==hashed(contract),'Stream seal differs')
        return rows

    def tail(self,name):
        """Validate a recovered history once, then verify its append cursor.

        Avoid reparsing all prior manifests on every future optimizer step.
        Full audit/restore still checks the entire immutable chain.
        """
        contract,seal=self.db.execute('SELECT contract,seal FROM streams WHERE name=?',(name,)).fetchone()
        require(seal is None,'Sealed stream cannot append')
        require(self.db.execute('SELECT 1 FROM intents WHERE name=?',(name,)).fetchone() is None,'Uncertain pending intent blocks automatic continuation')
        tail=self.db.execute('SELECT step,manifest,sha FROM frames WHERE name=? ORDER BY step DESC LIMIT 1',(name,)).fetchone()
        if name in self._heads:
            n,digest,last=self._heads[name]
            if (tail is None and n==0) or (tail is not None and (tail[0],tail[2])==(n-1,digest)):
                require(tail is None or hashed(tail[1])==digest,'Tail frame checksum differs')
                return n,last
        rows=self.frames(name,writable=True);n=len(rows)
        self._heads[name]=(n,None if tail is None else tail[2],None if not rows else rows[-1])
        return n,None if not rows else rows[-1]

    @contextmanager
    def transaction(self,name,step,intent,contract,*,fault=lambda phase:None):
        """Durably commit intent BEFORE caller update; never erase uncertainty.

        Caller computes its update, serializes complete state in memory, then
        commits all three original byte strings. One writer lease spans the unit.
        """
        require(re.fullmatch('[A-Za-z0-9_-]{1,100}',name) is not None,'Invalid stream identity')
        with self.locked():
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.execute('INSERT OR IGNORE INTO streams(name,contract) VALUES (?,?)',(name,contract))
                require(self.db.execute('SELECT contract FROM streams WHERE name=?',(name,)).fetchone()[0]==contract,'Frozen contract differs')
                count,last=self.tail(name);require(type(step) is int and step==count,'Next step differs')
                self.db.execute('INSERT INTO intents VALUES (?,?,?)',(name,step,intent));self.db.execute('COMMIT')
            except BaseException:
                if self.db.in_transaction:self.db.execute('ROLLBACK')
                raise
            fault('after_intent');tx=Transaction(self,name,step,last,fault)
            yield tx
            require(tx.committed,'No published frame; uncertain intent retained')

    def decode_file(self,frame,name):
        f=frame['files'][name];parts=[];offset=0
        require(0<f['bytes']<=MAX_FILE,'Decoded component outside size bound')
        for row in f['segments']:
            require(row['offset']==offset,'Segment gap/overlap');raw,_=self.get(row['object'])
            require(len(raw)==row['bytes'] and offset+len(raw)<=f['bytes'],'Segment length/size differs');parts.append(raw);offset+=len(raw)
        raw=b''.join(parts);require(len(raw)==f['bytes'] and hashed(raw)==f['sha256'],'Original file SHA differs')
        return raw

    def seal(self,name,original_done):
        with self.locked():
            self.db.execute('BEGIN IMMEDIATE')
            try:
                rows=self.frames(name,writable=True);require(bool(rows),'Empty stream')
                contract=self.db.execute('SELECT contract FROM streams WHERE name=?',(name,)).fetchone()[0]
                last=self.db.execute('SELECT sha FROM frames WHERE name=? ORDER BY step DESC LIMIT 1',(name,)).fetchone()[0]
                value={'schema':'c14_stream_seal_v2','frames':len(rows),'last_frame':last,'contract_sha256':hashed(contract),'original_done_hex':original_done.hex()}
                self.db.execute('UPDATE streams SET seal=? WHERE name=?',(encoded(value),name));self.db.execute('COMMIT')
            except BaseException:
                if self.db.in_transaction:self.db.execute('ROLLBACK')
                raise


class Transaction:
    def __init__(self,store,name,step,previous,fault):
        self.store,self.name,self.step,self.previous,self.fault=store,name,step,previous,fault;self.committed=False

    def commit(self,files):
        require(not self.committed and set(files)=={'state.pt','record.json','intent.json'},'Exactly one complete state/record/intent required')
        s=self.store;db=s.db;db.execute('BEGIN IMMEDIATE')
        try:
            require(db.execute('SELECT step,raw FROM intents WHERE name=?',(self.name,)).fetchone()==(self.step,files['intent.json']),'Durable intent changed')
            result={}
            for name,raw in files.items():
                base={} if self.previous is None else {r['slot']:r['object'] for r in self.previous['files'][name]['segments']}
                rows=[]
                for slot,offset,chunk in segments(raw,name,s.layout):
                    key=s.put(chunk,base.get(slot));rows.append({'slot':slot,'offset':offset,'bytes':len(chunk),'object':key})
                result[name]={'bytes':len(raw),'sha256':hashed(raw),'segments':rows}
            self.fault('after_objects')
            contract=db.execute('SELECT contract FROM streams WHERE name=?',(self.name,)).fetchone()[0]
            previous=None if self.step==0 else db.execute('SELECT sha FROM frames WHERE name=? AND step=?',(self.name,self.step-1)).fetchone()[0]
            value={'schema':'c14_frame_v2','step':self.step,'contract_sha256':hashed(contract),'previous':previous,'intent_sha256':hashed(files['intent.json']),'files':result}
            raw=encoded(value);db.execute('INSERT INTO frames VALUES (?,?,?,?,?)',(self.name,self.step,raw,hashed(raw),files['intent.json']))
            db.execute('DELETE FROM intents WHERE name=?',(self.name,))
            pages=db.execute('PRAGMA page_count').fetchone()[0];page_size=db.execute('PRAGMA page_size').fetchone()[0]
            require(2*pages*page_size+s.bytes()<=s.max_bytes,'Store reserve cap exceeded; intent retained')
            self.fault('before_publish');db.execute('COMMIT');self.committed=True
            s._heads[self.name]=(self.step+1,hashed(raw),value);self.fault('after_publish')
            return hashed(raw)
        except BaseException:
            if db.in_transaction:db.execute('ROLLBACK')
            raise
