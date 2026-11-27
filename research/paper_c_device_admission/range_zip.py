"""Bounded seekable HTTP Range reader; never silently download a full archive."""
import io
import hashlib
import re

import requests


class RangeFile(io.RawIOBase):
    def __init__(self,url,*,budget=16*1024**2,request_cap=16*1024**2):
        self.url=url;self.budget=budget;self.request_cap=request_cap;self.received=0;self.offset=0;self.ranges=[]
        with requests.head(url,allow_redirects=True,timeout=(15,30),headers={'Accept-Encoding':'identity'}) as r:
            r.raise_for_status();self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag')
        if self.size<=0 or not self.etag:raise ValueError('Version-bound archive size/ETag required')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.offset
    def seek(self,n,whence=0):
        target=n if whence==0 else self.offset+n if whence==1 else self.size+n if whence==2 else -1
        if not 0<=target<=self.size:raise ValueError('Invalid archive seek')
        self.offset=target;return target
    def read(self,n=-1):
        if n is None or n<0:n=self.size-self.offset
        n=min(n,self.size-self.offset)
        if n==0:return b''
        if n>self.request_cap or self.received+n>self.budget:raise ValueError('Range request/total budget exceeded before network read')
        start=self.offset;end=start+n-1
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','If-Match':self.etag}
        with requests.get(self.url,headers=headers,stream=True,timeout=(15,60)) as r:
            if r.status_code!=206:raise ValueError('Server ignored Range or source changed; full-body fallback forbidden')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            if not match or tuple(map(int,match.groups()))!=(start,end,self.size) or r.headers.get('ETag')!=self.etag:raise ValueError('Range/source identity differs')
            data=r.raw.read(n+1)
            if len(data)!=n:raise ValueError('Range byte count differs')
        self.received+=len(data);self.offset+=len(data);self.ranges.append({'start':start,'end':end,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        return data


def validate_member(info,*,cap):
    from pathlib import PurePosixPath
    path=PurePosixPath(info.filename)
    if path.is_absolute() or '..' in path.parts or '\\' in info.filename or info.flag_bits&1 or info.file_size>cap:
        raise ValueError('Unreviewed member path/encryption/decoded size')
