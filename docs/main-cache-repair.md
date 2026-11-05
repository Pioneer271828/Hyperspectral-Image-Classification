# Main VCA correctness repair — Issue #167

This candidate starts at main `4d2c7dbe4a081d671b1367bc4fb3eabf5e929ccc`.
It reuses the accepted adapter implementation from
`bb5a9cee98a781d52a544ea2bc678cf512e65576`, preserving default architecture,
parameter names and fresh training mathematics. Inference caching is bypassed
under autograd and invalidated on mode/state/device changes and normal in-place
weight updates. Returned diagnostics cannot mutate cached entries.

Remote acceptance uses the existing training container, never Windows Python:

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
REPAIR_DEVICE=cuda OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_vca_release_compatibility.py -v
python smoke_test.py --device cuda
```

The immutable-main oracle requires full Git history. This is a focused repair,
not a delivery of later graph fixes, transfer protocols or research mechanisms.
It must be reviewed before merging main. The separately tested development
integration provides those later modules; historical experiment tags stay fixed.
