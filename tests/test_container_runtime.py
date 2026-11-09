import unittest
from unittest.mock import patch
from transfer.container_runtime import training_container_kind


class ContainerRuntimeTests(unittest.TestCase):
    def test_explicit_markers_and_host_rejection(self):
        for platform, files, value, expected in (
            ('linux', {'/.dockerenv'}, '', 'docker'),
            ('linux', {'/run/systemd/container'}, 'lxc\n', 'lxc'),
            ('linux', {'/run/systemd/container'}, 'unknown', None),
            ('linux', set(), '', None), ('win32', {'/.dockerenv'}, '', None),
        ):
            with self.subTest(platform=platform, files=files, value=value), \
                 patch('transfer.container_runtime.sys.platform', platform), \
                 patch('transfer.container_runtime.Path.is_file', lambda path: str(path) in files), \
                 patch('transfer.container_runtime.Path.read_text', return_value=value):
                self.assertEqual(training_container_kind(), expected)
