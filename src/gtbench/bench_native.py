from __future__ import annotations

import time

import numpy as np

from . import native


def _np_median3(a):
    b = np.roll(a, 1, axis=1)
    c = np.roll(a, -1, axis=1)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.where(c < lo, lo, np.where(c > hi, hi, c))


def _numpy_combo(dl, dr, c0, c1, pred, gt, valid):
    _ = (np.abs(dl - dr) <= 1.0) & (dl > 0.5)
    _ = np.clip(1.0 - c0 / np.maximum(c1, 1e-6), 0.0, 1.0)
    m = valid.astype(bool) & (gt > 1e-6)
    return float((np.abs(np.maximum(pred[m], 1e-6) - gt[m]) / gt[m]).mean())


def benchmark(shape=(192, 320), repeats: int = 120) -> dict:
    """Compare C++17 kernels against equivalent NumPy expressions.

    The 3x1 median filter is the honest hot path: NumPy needs several
    temporaries, while the C++ kernel is a single pass over contiguous memory.
    """
    rng = np.random.default_rng(11)
    a = (rng.random(shape) * 20).astype(np.float32)
    dl = (rng.random(shape) * 48).astype(np.float32)
    dr = (dl + rng.normal(0, 0.5, shape)).astype(np.float32)
    c0 = rng.random(shape).astype(np.float32)
    c1 = (c0 + rng.random(shape) + 0.1).astype(np.float32)
    pred = np.maximum(a + rng.normal(0, 1, shape), 0.1).astype(np.float32)
    gt = np.maximum(a, 0.1).astype(np.float32)
    valid = (rng.random(shape) > 0.2).astype(np.uint8)

    def native_combo():
        native.lrc_filter(dl, dr, 1.0)
        native.confidence(c0, c1)
        return native.absrel(pred, gt, valid)

    for _ in range(3):
        native.median3(a)
        _np_median3(a)
        native_combo()
        _numpy_combo(dl, dr, c0, c1, pred, gt, valid)

    def timed(fn):
        t = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t) / repeats * 1000.0

    n_med = timed(lambda: native.median3(a))
    p_med = timed(lambda: _np_median3(a))
    n_combo = timed(native_combo)
    p_combo = timed(lambda: _numpy_combo(dl, dr, c0, c1, pred, gt, valid))

    return {
        "native_loaded": native.HAS_NATIVE,
        "shape": list(shape),
        "repeats": repeats,
        "median3": {
            "native_ms": round(n_med, 4),
            "numpy_ms": round(p_med, 4),
            "speedup": round(p_med / max(n_med, 1e-9), 2),
        },
        "combo": {
            "native_ms": round(n_combo, 4),
            "numpy_ms": round(p_combo, 4),
            "speedup": round(p_combo / max(n_combo, 1e-9), 2),
        },
    }
