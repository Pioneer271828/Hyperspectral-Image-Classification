"""Synthetic byte-stream recovery only, no model/data dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


def command(out,raw,digest=None):
    return [sys.executable,str(Path(__file__).with_name('receive_binary.py')),'--out',str(out),
            '--bytes',str(len(raw)),'--sha256',digest or hashlib.sha256(raw).hexdigest(),'--frame-bytes','32']


class BinaryRecovery(unittest.TestCase):
    def test_short_frame_not_written_then_exact_resume_and_readonly_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'archive.gz';raw=b'A'*32+b'B'*32+b'C'*7;cmd=command(out,raw)
            first=subprocess.run(cmd,input=raw[:40],capture_output=True)
            self.assertNotEqual(first.returncode,0);self.assertFalse(out.exists())
            part=out.with_suffix(out.suffix+'.partial');self.assertEqual(part.read_bytes(),raw[:32])
            status=json.loads(subprocess.check_output(cmd+['--status']));self.assertEqual(status['offset'],32)
            self.assertFalse(status['whole_source_agreement_verified'])
            final=subprocess.run(cmd,input=raw[32:],capture_output=True);self.assertEqual(final.returncode,0,final.stderr.decode())
            self.assertEqual(out.read_bytes(),raw);before=out.stat().st_mtime_ns
            self.assertTrue(json.loads(subprocess.check_output(cmd+['--status']))['complete']);self.assertEqual(out.stat().st_mtime_ns,before)

    def test_corrupt_committed_prefix_and_orphan_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'archive';raw=b'A'*64;cmd=command(out,raw)
            subprocess.run(cmd,input=raw[:32],capture_output=True)
            part=out.with_suffix('.partial');part.write_bytes(b'X'*32)
            before=part.read_bytes();bad=subprocess.run(cmd,input=raw[32:],capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertEqual(part.read_bytes(),before);self.assertFalse(out.exists())
            orphan=Path(tmp)/'orphan';orphan.with_suffix('.partial').write_bytes(b'KEEP')
            bad=subprocess.run(command(orphan,b'new'),input=b'new',capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertEqual(orphan.with_suffix('.partial').read_bytes(),b'KEEP')

    def test_complete_wrong_source_hash_never_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'wrong';raw=b'SYNTHETIC-BYTES'*8
            bad=subprocess.run(command(out,raw,'0'*64),input=raw,capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(out.exists())
            self.assertEqual(out.with_suffix('.partial').read_bytes(),raw)

    def test_second_writer_rejected_while_first_owns_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'locked';raw=b'A'*64;cmd=command(out,raw)
            first=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if out.with_suffix('.partial').exists():break
                    time.sleep(.01)
                self.assertTrue(out.with_suffix('.partial').exists())
                second=subprocess.run(cmd+['--status'],capture_output=True)
                self.assertNotEqual(second.returncode,0)
                _,err=first.communicate(raw,timeout=5);self.assertEqual(first.returncode,0,err.decode());self.assertEqual(out.read_bytes(),raw)
            finally:
                if first.poll() is None:first.kill();first.communicate()


if __name__=='__main__':unittest.main()
