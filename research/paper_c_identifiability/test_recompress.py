"""Byte-exact lossless archive transformation on synthetic content only."""
import gzip
import hashlib
import io
import lzma
from pathlib import Path
import tarfile
import tempfile
import unittest

from recompress import encode


class LosslessArchive(unittest.TestCase):
    def test_whole_tar_bytes_and_member_hashes_survive_reencoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.tar.gz';raw=io.BytesIO();payload=b'SYNTHETIC-OPAQUE-BYTES'*1000
            with tarfile.open(fileobj=raw,mode='w') as t:
                info=tarfile.TarInfo('object');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
            source.write_bytes(gzip.compress(raw.getvalue()));before=source.read_bytes();out=root/'new.tar.xz'
            receipt=encode(source,out,hashlib.sha256(before).hexdigest())
            self.assertEqual(lzma.decompress(out.read_bytes()),raw.getvalue());self.assertEqual(source.read_bytes(),before)
            with tarfile.open(out,'r:*') as t:self.assertEqual(t.extractfile('object').read(),payload)
            self.assertEqual(receipt['tensor_or_observation_decodes'],0)
            with self.assertRaises(ValueError):encode(source,out,hashlib.sha256(before).hexdigest())


if __name__=='__main__':unittest.main()
