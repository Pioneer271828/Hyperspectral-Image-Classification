"""Exercise archive integrity and safe recovery boundaries on synthetic bytes."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from recover import relative, restore, sha


class RecoveryTests(unittest.TestCase):
    def archive(self, root, *, direct=False, corrupt=False, missing=False):
        raw = b'frozen synthetic evidence'
        key = hashlib.sha256(raw).hexdigest()
        pin = {'bytes': len(raw), 'sha256': key}
        index = {'files': {'case/state.bin': pin}}
        if not direct:
            index['objects'] = {key: {'bytes': len(raw)}}
        path = root / 'test.tar.gz'
        with tarfile.open(path, 'x:gz') as tar:
            def add(name, data):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            add('index.json', json.dumps(index).encode())
            if not missing:
                add('case/state.bin' if direct else 'objects/' + key,
                    b'X' * len(raw) if corrupt else raw)
        return path, raw

    def test_deduplicated_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, raw = self.archive(root)
            restore(archive, archive.stat().st_size, sha(archive), root / 'restore')
            self.assertEqual((root / 'restore/restored/case/state.bin').read_bytes(), raw)

    def test_direct_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, raw = self.archive(root, direct=True)
            restore(archive, archive.stat().st_size, sha(archive), root / 'restore')
            self.assertEqual((root / 'restore/restored/case/state.bin').read_bytes(), raw)

    def test_corrupt_or_missing_object_rejected(self):
        for option in ('corrupt', 'missing'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive, _ = self.archive(root, **{option: True})
                with self.assertRaises(ValueError):
                    restore(archive, archive.stat().st_size, sha(archive), root / 'restore')

    def test_unsafe_paths_rejected(self):
        for name in ('../outside', '/outside', 'C:/outside', 'x\\outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(name)


if __name__ == '__main__':
    unittest.main()
