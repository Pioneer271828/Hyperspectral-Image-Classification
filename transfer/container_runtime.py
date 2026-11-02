"""Recognize explicit Linux container markers without accepting a bare host."""
from pathlib import Path
import sys


def training_container_kind():
    if sys.platform != 'linux':
        return None
    if Path('/.dockerenv').is_file():
        return 'docker'
    marker = Path('/run/systemd/container')
    if marker.is_file() and marker.read_text().strip() in ('lxc', 'lxc-libvirt'):
        return 'lxc'
    return None
