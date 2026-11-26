"""Mixed durable frame sizes and multi-GiB numeric contracts, synthetic bytes."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def frame(offset,raw):return f'{offset}|{hashlib.sha256(raw).hexdigest()}|'.encode()+base64.b64encode(raw)+b'\n'


class LargerFrames(unittest.TestCase):
    def test_resume_one_MiB_then_large_frame_and_final_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'archive';raw=b'SYNTHETIC-BACKUP-'*200000;digest=hashlib.sha256(raw).hexdigest()
            cmd=[sys.executable,str(Path(__file__).with_name('receive_backup_stream.py')),'--out',str(out),'--bytes',str(len(raw)),'--sha256',digest]
            first=subprocess.run(cmd,input=frame(0,raw[:1048576]),capture_output=True)
            self.assertNotEqual(first.returncode,0)
            status=json.loads(subprocess.check_output(cmd+['--status']));self.assertEqual(status['offset'],1048576)
            finish=subprocess.run(cmd+['--frame-bytes','16777216'],input=frame(1048576,raw[1048576:])+b'END\n',capture_output=True)
            self.assertEqual(finish.returncode,0,finish.stderr.decode());self.assertEqual(out.read_bytes(),raw)
            before=out.stat().st_mtime_ns
            again=json.loads(subprocess.check_output(cmd));self.assertTrue(again['complete']);self.assertEqual(out.stat().st_mtime_ns,before)

    def test_large_numeric_archive_and_bad_large_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'large';cmd=[sys.executable,str(Path(__file__).with_name('receive_backup_stream.py')),'--out',str(out),'--bytes','8337564720','--sha256','0'*64]
            bad=subprocess.run(cmd,input=frame(0,b'x'*(2*1048576)),capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(out.exists())
            self.assertEqual(out.with_suffix('.partial').stat().st_size,0)


if __name__=='__main__':unittest.main()
