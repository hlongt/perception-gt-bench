from __future__ import annotations

import numpy as np

from ..geometry import box_sum
from .base import DepthProposal


def _cost_volume(ref: np.ndarray, target: np.ndarray, dmax: int, radius: int, mode: str):
    """(dmax+1, H, W) SAD cost volume over rectified scanlines.

    mode='left' : target[u-d] is compared against ref[u] (left-view reference).
    mode='right': target[u+d] is compared against ref[u] (right-view reference).
    """
    h, w = ref.shape
    vol = np.empty((dmax + 1, h, w), dtype=np.float32)
    uu = np.arange(w)
    for d in range(dmax + 1):
        if mode == "left":
            shifted = np.pad(target, ((0, 0), (d, 0)), mode="edge")[:, :w]
            invalid = uu[None, :] < d
        else:
            shifted = np.pad(target, ((0, 0), (0, d)), mode="edge")[:, d : d + w]
            invalid = uu[None, :] >= (w - d)
        cost = box_sum(np.abs(ref - shifted), radius).astype(np.float32)
        cost = np.broadcast_to(cost, (h, w)).copy()
        cost[:, invalid.ravel()] = 1e6
        vol[d] = cost
    return vol


def _wta(vol: np.ndarray, subpixel: bool = True):
    dmax = vol.shape[0] - 1
    best = np.argmin(vol, axis=0).astype(np.float32)
    idx = best.astype(np.int32)[None]
    c0 = np.take_along_axis(vol, idx, axis=0)[0]
    cm = np.take_along_axis(vol, np.clip(idx - 1, 0, dmax), axis=0)[0]
    cp = np.take_along_axis(vol, np.clip(idx + 1, 0, dmax), axis=0)[0]
    vol2 = vol.copy()
    np.put_along_axis(vol2, idx, np.inf, axis=0)
    c1 = np.min(vol2, axis=0)
    denom = cm - 2.0 * c0 + cp
    delta = np.where(np.abs(denom) > 1e-6, 0.5 * (cm - cp) / np.where(np.abs(denom) > 1e-6, denom, 1.0), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    disp = np.clip(best + (delta if subpixel else 0.0), 0.0, float(dmax))
    peak = np.clip(1.0 - c0 / np.maximum(c1, 1e-6), 0.0, 1.0)
    return disp, peak


def compute_disparity(gray_left: np.ndarray, gray_right: np.ndarray, cfg):
    dmax = int(cfg.max_disparity)
    radius = max(1, int(cfg.window) // 2)
    vl = _cost_volume(gray_left, gray_right, dmax, radius, "left")
    vr = _cost_volume(gray_right, gray_left, dmax, radius, "right")
    disp_l, peak_l = _wta(vl, bool(cfg.subpixel))
    disp_r, _ = _wta(vr, bool(cfg.subpixel))

    h, w = disp_l.shape
    uu = np.broadcast_to(np.arange(w)[None, :], (h, w)).astype(np.float32)
    gather = np.clip(np.rint(uu - disp_l).astype(np.int32), 0, w - 1)
    disp_r_at_l = np.take_along_axis(disp_r, gather, axis=1)
    lr_ok = np.abs(disp_l - disp_r_at_l) <= float(cfg.lr_threshold)
    lr_margin = np.clip(1.0 - np.abs(disp_l - disp_r_at_l) / (float(cfg.lr_threshold) * 3.0), 0.0, 1.0)

    conf = np.clip(0.6 * peak_l + 0.4 * lr_margin, 0.0, 1.0)
    valid = lr_ok & (disp_l > 0.5) & (conf >= float(cfg.min_confidence))
    return disp_l, conf, valid


def stereo_proposal(frame, cam, cfg) -> DepthProposal:
    """Dense metric depth from rectified stereo matching.

    Implements SAD window matching, winner-take-all, sub-pixel parabola fit,
    left-right consistency and peak-ratio confidence from scratch (no OpenCV).
    """
    disp, conf, valid = compute_disparity(frame.gray_left, frame.gray_right, cfg)
    depth = cam.depth_from_disparity(np.maximum(disp, 1e-3)).astype(np.float32)
    depth = np.clip(depth, 0.1, 100.0)
    return DepthProposal("stereo", depth, conf, valid)
