"""Synthetic corruption and committed-boundary recovery; no SSH/NN/data access."""
import base64,hashlib,io,json,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).parent
def sha(raw):return hashlib.sha256(raw).hexdigest()


class BackupProtocolTests(unittest.TestCase):
    def run_receiver(self,out,raw,wire=b'',status=False):
        cmd=[sys.executable,str(ROOT/'receive_backup_stream.py'),'--out',str(out),
             '--bytes',str(len(raw)),'--sha256',sha(raw)]
        if status:cmd.append('--status')
        return subprocess.run(cmd,input=wire,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    def frame(self,offset,raw):return str(offset).encode()+b'|'+sha(raw).encode()+b'|'+base64.b64encode(raw)+b'\n'

    def test_committed_chunk_resumes_and_completed_reentry_is_read_only(self):
        raw=b'c'*1048576+b'final-boundary'
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'archive.bin'
            first=self.run_receiver(out,raw,self.frame(0,raw[:1048576])+b'END\n')
            self.assertNotEqual(first.returncode,0);self.assertFalse(out.exists())
            state=self.run_receiver(out,raw,status=True)
            self.assertEqual(json.loads(state.stdout)['offset'],1048576)
            second=self.run_receiver(out,raw,self.frame(1048576,raw[1048576:])+b'END\n')
            self.assertEqual(second.returncode,0,second.stderr.decode())
            self.assertEqual(out.read_bytes(),raw)
            before=out.stat().st_mtime_ns
            done=self.run_receiver(out,raw,status=True)
            self.assertTrue(json.loads(done.stdout)['complete']);self.assertEqual(out.stat().st_mtime_ns,before)

    def test_corrupt_committed_chunk_and_bad_frame_are_rejected_without_overwrite(self):
        raw=b'd'*1048576+b'tail'
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'archive.bin'
            self.run_receiver(out,raw,self.frame(0,raw[:1048576])+b'END\n')
            partial=out.with_suffix('.bin.partial');partial.write_bytes(b'x'+partial.read_bytes()[1:])
            damaged=partial.read_bytes();failed=self.run_receiver(out,raw,status=True)
            self.assertNotEqual(failed.returncode,0);self.assertEqual(partial.read_bytes(),damaged)
            other=Path(tmp)/'bad.bin'
            bad=self.run_receiver(other,b'abc',b'0|'+b'0'*64+b'|'+base64.b64encode(b'abc')+b'\n')
            self.assertNotEqual(bad.returncode,0);self.assertFalse(other.exists())

    def test_object_restore_retains_duplicate_logical_paths_and_rejects_corruption(self):
        raw=b'synthetic duplicate';h=sha(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'objects.tar.gz'
            index={'files':[{'name':'one/result.bin','bytes':len(raw),'sha256':h},
                            {'name':'two/result.bin','bytes':len(raw),'sha256':h}],
                   'objects':{h:{'bytes':len(raw)}}}
            def make(path,payload):
                with tarfile.open(path,'w:gz') as t:
                    for name,value in [('objects/'+h,payload),('object-index01.json',json.dumps(index).encode())]:
                        item=tarfile.TarInfo(name);item.size=len(value);t.addfile(item,io.BytesIO(value))
            make(archive,raw)
            args=[sys.executable,str(ROOT/'restore_backup_objects.py'),'--archive',str(archive)]
            done=subprocess.run(args+['--out',str(root/'valid'),'--receipt',str(root/'receipt.json')],capture_output=True)
            self.assertEqual(done.returncode,0,done.stderr.decode())
            self.assertEqual((root/'valid/restored/one/result.bin').read_bytes(),raw)
            self.assertEqual((root/'valid/restored/two/result.bin').read_bytes(),raw)
            corrupt=root/'corrupt.tar.gz';make(corrupt,b'wrong')
            failed=subprocess.run([sys.executable,str(ROOT/'restore_backup_objects.py'),'--archive',str(corrupt),
                                   '--out',str(root/'bad'),'--receipt',str(root/'bad-receipt.json')],capture_output=True)
            self.assertNotEqual(failed.returncode,0);self.assertFalse((root/'bad-receipt.json').exists())


if __name__=='__main__':unittest.main()
