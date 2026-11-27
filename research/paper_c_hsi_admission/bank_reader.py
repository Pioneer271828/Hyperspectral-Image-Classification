"""Read existing released HSI blocks without a raw-GT, full-scene, or model loader."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np


PINS={
    'xuzhou':{'bank':'c877232e6cc38e3f8be37eb1b147d69a7253edbeb665f98e5b8cd6acb8e5cb17','registration':'bc555ce89034dafa4ca99602ba067f8386a685d7528e4eab27793bfc142a6104','shape':(500,260,436),'classes':9},
    'whu-honghu':{'bank':'0bbe03fc60abe505c2c48d2f4837ba7494fabbcdeda673280d7220d5bc13dfa9','registration':'77876433efa118908a90571d0b4124a3e0d0f77b0eb930adfa8ae22a1052d2be','shape':(940,475,270),'classes':22}
}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def safe_member(root,name):
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('Original relative bank member required')
    path=root/name
    if path.is_symlink() or path.resolve().parent!=root.resolve():raise ValueError('Bank member escapes its immutable root')
    return path


def pinned_json(path,expected,*,cap=4*1024**2):
    if path.is_symlink() or path.stat().st_size>cap:raise ValueError('Unreviewed metadata alias/size')
    raw=path.read_bytes()
    if sha(raw)!=expected:raise ValueError('Pinned metadata bytes differ')
    return json.loads(raw)


class ReleasedBank:
    """Authenticates authority and samples before any spectral-array payload read.

    Storage root may be a private restored copy. Original JSON bytes and their
    old absolute provenance paths remain unchanged; no path is rewritten in-place.
    """
    def __init__(self,root,scene,label_release):
        if scene not in PINS:raise ValueError('Scene has no registered original bank pin')
        self.root=Path(root).resolve();self.scene=scene;self.pin=PINS[scene];self.payload_reads=0;self.payload_bytes=0
        self.registration=pinned_json(safe_member(self.root,'registration.json'),self.pin['registration'])
        self.index=pinned_json(safe_member(self.root,'bank.json'),self.pin['bank'])
        reg=self.registration
        if tuple(reg['shape'])!=self.pin['shape'] or reg['classes']!=self.pin['classes'] or reg['scene']!=scene or self.index['registration']!=self.pin['registration']:raise ValueError('Original scene shape/class authority differs')
        label_pin=reg['custody']['label_release'];self.label_release=pinned_json(Path(label_release),label_pin['sha256'])
        labels=self.label_release
        if labels['scene']!=scene or set(labels)!={'scene','development_roles_hash','support','validation','release_hash'}:raise ValueError('Only the original two-role K5 release is admitted')
        self.rows={r['id']:r for r in reg['rows'] if r['blocked'] is None};self.samples={};self.members={};seen=set();role_points={}
        for role,key in [('devtrain','support'),('devval','validation')]:
            group=labels[key]
            if group['per_class']!=5 or len(group['indices'])!=len(group['labels']):raise ValueError('Original K5/V5 label budget differs')
            points=list(zip(group['indices'],group['labels']))
            if len({x for x,_ in points})!=len(points) or set(x for x,_ in points)&seen:raise ValueError('Repeated original point identity')
            seen.update(x for x,_ in points);counts={c:sum(y==c for _,y in points) for c in range(1,self.pin['classes']+1)}
            if set(counts.values())!={5}:raise ValueError('Original five-per-class memberships differ')
            role_points[role]=points
        for item in self.index['members']:
            row=self.rows[item['id']]
            if item['role']!=row['role'] or item['name']!=f"tile-{row['id']}.json":raise ValueError('Bank member role/name differs')
            meta=pinned_json(safe_member(self.root,item['name']),item['sha256'])
            if meta['row']!=row or meta['registration']!=self.pin['registration'] or meta['member']!=f"tile-{row['id']}.npy" or meta['dtype']!='<f4':raise ValueError('Tile manifest differs from original authority')
            y0,y1,x0,x1=row['box'];width=self.pin['shape'][1]
            points=sorted([[int(i),int(c)] for i,c in role_points[row['role']] if y0<=i//width<y1 and x0<=i%width<x1])
            if row['released_samples']!={'count':len(points),'sha256':sha(canonical(points))}:raise ValueError('Original row sample digest differs')
            self.samples[row['id']]=points;self.members[row['id']]=meta
        if set(self.members)!=set(self.rows):raise ValueError('Original completed bank roster differs')
        for role,points in role_points.items():
            gathered=[p for i,r in self.rows.items() if r['role']==role for p in self.samples[i]]
            if sorted(gathered)!=sorted([list(p) for p in points]):raise ValueError('Original released labels are not covered exactly once')

    def role_rows(self,role):
        if role not in ('devtrain','devval'):raise ValueError('Only original released development roles admitted')
        return tuple(r for r in self.rows.values() if r['role']==role)

    def read_tile(self,role,tile_id):
        if role not in ('devtrain','devval') or type(tile_id)!=int or tile_id not in self.rows or self.rows[tile_id]['role']!=role:raise ValueError('Requested role/tile was not released')
        row=self.rows[tile_id]
        if not self.samples[tile_id]:raise ValueError('This intake admits only originally labeled tiles, no observation expansion')
        # Reauthenticate the two authority roots immediately before each read.
        pinned_json(safe_member(self.root,'registration.json'),self.pin['registration']);pinned_json(safe_member(self.root,'bank.json'),self.pin['bank'])
        meta=self.members[tile_id];path=safe_member(self.root,meta['member']);content=meta['content']
        if path.stat().st_size!=content['bytes']:raise ValueError('Original tile byte count differs')
        raw=path.read_bytes()
        if sha(raw)!=content['npy_sha256']:raise ValueError('Original NPY bytes differ')
        stream=io.BytesIO(raw)
        if np.lib.format.read_magic(stream)!=(1,0):raise ValueError('Original bank NPY v1 required')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096);y0,y1,x0,x1=row['box'];expected=(y1-y0,x1-x0,self.pin['shape'][2])
        if shape!=expected or fortran or dtype!=np.dtype('<f4') or len(raw)-stream.tell()!=np.prod(shape)*4:raise ValueError('Tile header escapes its original context')
        stream.seek(0);cube=np.load(stream,allow_pickle=False)
        if not np.isfinite(cube).all():raise ValueError('Nonfinite released spectral values')
        stats=self.registration['statistics'];low=np.asarray(stats['minimum'],np.float32);high=np.asarray(stats['maximum'],np.float32)
        if row['representation']=='raw_fp32':normalized=(cube-low)/np.where(high>low,high-low,1)
        elif row['representation']=='normalized_fp32':normalized=cube
        else:raise ValueError('Unknown released representation')
        cube.setflags(write=False);normalized.setflags(write=False);self.payload_reads+=1;self.payload_bytes+=len(raw)
        return {'spectral_raw_hwc':cube,'spectral_normalized_hwc':normalized,'samples':tuple(tuple(x) for x in self.samples[tile_id]),'role':role,'tile_id':tile_id,'original_box':tuple(row['box']),'parent':row['parent_tile'],'processed_column_ids':tuple(range(shape[2])),'native_physical_mapping_authenticated':False,'normalization':'Original frozen bank training-only statistics, no refit','npy_sha256':content['npy_sha256']}
