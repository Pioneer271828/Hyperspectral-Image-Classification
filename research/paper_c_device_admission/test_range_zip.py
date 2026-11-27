"""Actual HTTP behavior: ignored ranges, version changes and bounded ZIP reads."""
import http.server
import io
import threading
import unittest
import zipfile

from range_zip import RangeFile,validate_member


class RangeProtocol(unittest.TestCase):
    def setUp(self):
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('D006_2020/IMG_example.tif',b'SYNTHETIC-PUBLIC-TRAIN'*100)
        self.raw=f.getvalue();self.ignore=False;self.changed=False;owner=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_HEAD(self):
                self.send_response(200);self.send_header('Content-Length',str(len(owner.raw)));self.send_header('ETag','"original"');self.end_headers()
            def do_GET(self):
                start,end=map(int,self.headers['Range'][6:].split('-'));body=owner.raw[start:end+1]
                self.send_response(200 if owner.ignore else 206);self.send_header('Content-Length',str(len(body)))
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(owner.raw)}');self.send_header('ETag','"changed"' if owner.changed else '"original"');self.end_headers();self.wfile.write(body)
        self.server=http.server.HTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url=f'http://127.0.0.1:{self.server.server_port}/train.zip'
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def test_zip_directory_member_crc_and_size_budget(self):
        remote=RangeFile(self.url,budget=4096)
        with zipfile.ZipFile(remote) as z:
            item=z.infolist()[0];validate_member(item,cap=4096);self.assertEqual(z.read(item),b'SYNTHETIC-PUBLIC-TRAIN'*100)
        with self.assertRaises(ValueError):RangeFile(self.url,budget=1).read(2)
    def test_ignored_range_rejected_without_full_body_fallback(self):
        remote=RangeFile(self.url);self.ignore=True
        with self.assertRaises(ValueError):remote.read(10)
        self.assertEqual(remote.received,0)
    def test_changed_version_and_unsafe_member_rejected(self):
        remote=RangeFile(self.url);self.changed=True
        with self.assertRaises(ValueError):remote.read(10)
        with self.assertRaises(ValueError):validate_member(zipfile.ZipInfo('../escape'),cap=10)


if __name__=='__main__':unittest.main()
