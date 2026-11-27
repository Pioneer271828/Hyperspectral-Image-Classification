"""Opt-in SLIC assignment on an already admitted, normalized HSI block."""
import numpy as np
from scipy.sparse import csr_matrix


def slic_assignment(cube, *, scale=None, regions=None):
    """Return canonical labels and one sparse Q entry per input pixel."""
    from skimage.segmentation import slic

    if (not isinstance(cube, np.ndarray) or cube.dtype != np.float32 or cube.ndim != 3
            or not all(cube.shape) or not np.isfinite(cube).all()):
        raise ValueError("Expected a finite FP32 HSI block in HWC order")
    pixels = cube.shape[0] * cube.shape[1]
    if regions is None:
        if type(scale) is not int or scale < 1:
            raise ValueError("Positive integer region scale required")
        requested = max(1, pixels // scale)
    else:
        if scale is not None or type(regions) is not int or not 1 <= regions <= pixels:
            raise ValueError("Use one explicit region budget between one and the pixel count")
        requested = regions
    labels = slic(cube, n_segments=requested, compactness=0.5, max_num_iter=20, sigma=1,
                  convert2lab=False, enforce_connectivity=True, min_size_factor=0.1,
                  max_size_factor=2, slic_zero=False, start_label=0, channel_axis=-1)
    if labels.shape != cube.shape[:2] or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("SLIC returned invalid assignment labels")
    # Region names are arbitrary. Assign columns by first pixel in raster order.
    _, first, inverse = np.unique(labels.ravel(), return_index=True, return_inverse=True)
    order = np.argsort(first)
    columns = np.empty(len(order), dtype=np.int64)
    columns[order] = np.arange(len(order))
    canonical = columns[inverse]
    assignment = csr_matrix((np.ones(pixels, dtype=np.float32), canonical, np.arange(pixels + 1)),
                            shape=(pixels, len(order)))
    return canonical.reshape(labels.shape), assignment
