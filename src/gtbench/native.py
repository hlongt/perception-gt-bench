from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_SUFFIX = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
_LIB_PATH = Path(__file__).with_name("_native" + _SUFFIX)
_lib = None
HAS_NATIVE = False

_PF = ctypes.POINTER(ctypes.c_float)
_PU8 = ctypes.POINTER(ctypes.c_ubyte)
_PLONG = ctypes.POINTER(ctypes.c_longlong)


def _load():
    global _lib, HAS_NATIVE
    if _lib is not None or not _LIB_PATH.exists():
        return
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.gtb_lrc_filter.argtypes = [_PF, _PF, _PU8, ctypes.c_int, ctypes.c_float]
        lib.gtb_lrc_filter.restype = None
        lib.gtb_confidence.argtypes = [_PF, _PF, _PF, ctypes.c_int]
        lib.gtb_confidence.restype = None
        lib.gtb_median3.argtypes = [_PF, _PF, ctypes.c_int, ctypes.c_int]
        lib.gtb_median3.restype = None
        lib.gtb_absrel.argtypes = [_PF, _PF, _PU8, ctypes.c_int]
        lib.gtb_absrel.restype = ctypes.c_double
        lib.gtb_delta_counts.argtypes = [
            _PF, _PF, _PU8, ctypes.c_int, ctypes.c_double, _PLONG, _PLONG
        ]
        lib.gtb_delta_counts.restype = None
        _lib = lib
        HAS_NATIVE = True
    except OSError:
        _lib = None


def _f(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _u8(a):
    return np.ascontiguousarray(a, dtype=np.uint8)


def lrc_filter(dl, dr_gathered, thresh: float = 1.0) -> np.ndarray:
    _load()
    dl, dr = _f(dl), _f(dr_gathered)
    out = np.zeros(dl.shape, dtype=np.uint8)
    if HAS_NATIVE:
        _lib.gtb_lrc_filter(
            dl.ctypes.data_as(_PF), dr.ctypes.data_as(_PF),
            out.ctypes.data_as(_PU8), dl.size, float(thresh),
        )
    else:
        out[:] = ((np.abs(dl - dr) <= thresh) & (dl > 0.5)).astype(np.uint8)
    return out.astype(bool)


def confidence(c0, c1) -> np.ndarray:
    _load()
    c0, c1 = _f(c0), _f(c1)
    out = np.zeros(c0.shape, dtype=np.float32)
    if HAS_NATIVE:
        _lib.gtb_confidence(
            c0.ctypes.data_as(_PF), c1.ctypes.data_as(_PF),
            out.ctypes.data_as(_PF), c0.size,
        )
    else:
        out[:] = np.clip(1.0 - c0 / np.maximum(c1, 1e-6), 0.0, 1.0)
    return out


def absrel(pred, gt, valid) -> float:
    _load()
    pred, gt, valid = _f(pred), _f(gt), _u8(valid)
    if HAS_NATIVE:
        return float(_lib.gtb_absrel(
            pred.ctypes.data_as(_PF), gt.ctypes.data_as(_PF),
            valid.ctypes.data_as(_PU8), pred.size,
        ))
    m = valid.astype(bool) & (gt > 1e-6)
    return float((np.abs(np.maximum(pred[m], 1e-6) - gt[m]) / gt[m]).mean()) if m.any() else 0.0


def delta_accuracy(pred, gt, valid, thr: float = 1.25) -> float:
    _load()
    pred, gt, valid = _f(pred), _f(gt), _u8(valid)
    if HAS_NATIVE:
        hit = ctypes.c_longlong(0)
        total = ctypes.c_longlong(0)
        _lib.gtb_delta_counts(
            pred.ctypes.data_as(_PF), gt.ctypes.data_as(_PF),
            valid.ctypes.data_as(_PU8), pred.size, float(thr),
            ctypes.byref(hit), ctypes.byref(total),
        )
        return float(hit.value) / total.value if total.value else 0.0
    m = valid.astype(bool) & (gt > 1e-6)
    p, g = np.maximum(pred[m], 1e-6), gt[m]
    return float((np.maximum(p / g, g / p) < thr).mean()) if len(g) else 0.0


def median3(img) -> np.ndarray:
    _load()
    a = _f(img)
    out = np.zeros_like(a)
    if HAS_NATIVE:
        h, w = a.shape
        _lib.gtb_median3(
            a.ctypes.data_as(_PF), out.ctypes.data_as(_PF), h, w
        )
    else:
        out[:] = np.minimum(np.maximum(np.minimum(a, np.roll(a, 1, 1)), np.roll(a, -1, 1)),
                            np.maximum(a, np.minimum(np.roll(a, 1, 1), np.roll(a, -1, 1))))
    return out


def status() -> dict:
    _load()
    return {"native_loaded": HAS_NATIVE, "library": str(_LIB_PATH), "exists": _LIB_PATH.exists()}
