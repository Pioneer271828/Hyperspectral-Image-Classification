"""Native-forwarding endpoints use private sockets and byte-exact source pins."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class PrivateSocket(unittest.TestCase):
    def test_pinned_one_shot_source_and_receiver_match_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'source.gz';raw=b'SYNTHETIC-SOCKET-BYTES'*100000;archive.write_bytes(raw)
            endpoint=root/'stream.sock';out=root/'received.gz';digest=hashlib.sha256(raw).hexdigest();folder=Path(__file__).parent
            server=subprocess.Popen([sys.executable,str(folder/'serve_socket.py'),'--archive',str(archive),'--socket',str(endpoint),
                '--bytes',str(len(raw)),'--sha256',digest,'--offset','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                for _ in range(300):
                    if endpoint.exists():break
                    time.sleep(.01)
                self.assertTrue(endpoint.exists());self.assertEqual(endpoint.stat().st_mode&0o777,0o600)
                client=subprocess.run([sys.executable,str(folder/'receive_binary.py'),'--out',str(out),'--bytes',str(len(raw)),
                    '--sha256',digest,'--socket',str(endpoint)],capture_output=True,timeout=10)
                self.assertEqual(client.returncode,0,client.stderr.decode());_,err=server.communicate(timeout=10)
                self.assertEqual(server.returncode,0,err.decode());self.assertEqual(out.read_bytes(),raw)
            finally:
                if server.poll() is None:server.kill();server.communicate()


if __name__=='__main__':unittest.main()
