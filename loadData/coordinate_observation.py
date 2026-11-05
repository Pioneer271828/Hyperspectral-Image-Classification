"""Label-blind coordinate representation; spectral data and global IDs stay intact."""
from dataclasses import replace
import numpy as np

MODES = ('absolute', 'relative', 'zero')

def coordinate_observation(observation, mode='absolute'):
    if mode not in MODES:
        raise ValueError('Unknown coordinate_mode')
    if mode == 'absolute': return observation
    h, w = observation.spectral.shape[:2]
    coordinates = np.zeros((h, w, 2), dtype=np.float32)
    if mode == 'relative':
        coordinates[..., 0] = np.arange(h, dtype=np.float32)[:, None] / max(h-1, 1)
        coordinates[..., 1] = np.arange(w, dtype=np.float32)[None, :] / max(w-1, 1)
    coordinates.setflags(write=False)
    return replace(observation, coordinates=coordinates)
