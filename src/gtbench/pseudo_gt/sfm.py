from __future__ import annotations

import numpy as np

from ..geometry import (
    box_sum,
    epipolar_residual,
    essential_matrix,
    harris_corners,
    reprojection_error,
    triangulate_dlt,
)
from .base import DepthProposal

_P = 7  # keypoint descriptor side length


def _match_keypoints(gray_l, gray_r, kps, dmax, radius):
    """Vectorised per-keypoint scanline SAD match; returns (d, peak, coords)."""
    if len(kps) == 0:
        return np.zeros((0,)), np.zeros((0,)), kps
    pad_l = np.pad(gray_l, ((radius, radius), (radius, radius)), mode="edge")
    pad_r = np.pad(gray_r, ((radius, radius), (radius, radius)), mode="edge")
    us = np.clip(np.rint(kps[:, 0]).astype(int), radius, gray_l.shape[1] - radius - 1)
    vs = np.clip(np.rint(kps[:, 1]).astype(int), radius, gray_l.shape[0] - radius - 1)
    patches = np.stack(
        [pad_l[v : v + _P, u : u + _P] for u, v in zip(us, vs)]
    ).astype(np.float32)
    costs = np.full((len(us), dmax + 1), np.inf, dtype=np.float32)
    for d in range(1, dmax + 1):
        ok = (us - d) >= radius
        if not ok.any():
            break
        cand = np.stack(
            [
                pad_r[v : v + _P, u - d : u - d + _P] if flag else np.zeros_like(patches[0])
                for (u, v), flag in zip(zip(us, vs), ok)
            ]
        ).astype(np.float32)
        c = np.abs(patches - cand).mean(axis=(1, 2))
        costs[ok, d] = c[ok]
    best = np.argmin(costs, axis=1)
    c0 = costs[np.arange(len(best)), best]
    tmp = costs.copy()
    tmp[np.arange(len(best)), best] = np.inf
    c1 = tmp.min(axis=1)
    peak = np.clip(1.0 - c0 / np.maximum(c1, 1e-6), 0.0, 1.0)
    peak = np.nan_to_num(peak, nan=0.0, posinf=0.0, neginf=0.0)
    return best.astype(np.float32), peak, np.column_stack([us, vs]).astype(np.float64)


def _scatter_grid(depth_map, conf_map, h, w, cell=16):
    """Diffuse sparse SfM points into a coarse, smoothed dense prior."""
    gh, gw = int(np.ceil(h / cell)), int(np.ceil(w / cell))
    acc = np.zeros((gh, gw), np.float64)
    cnt = np.zeros((gh, gw), np.float64)
    cacc = np.zeros((gh, gw), np.float64)
    ys, xs = np.nonzero(conf_map > 0)
    if len(xs) == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    gy = np.clip(ys // cell, 0, gh - 1)
    gx = np.clip(xs // cell, 0, gw - 1)
    np.add.at(acc, (gy, gx), depth_map[ys, xs])
    np.add.at(cnt, (gy, gx), 1.0)
    np.add.at(cacc, (gy, gx), conf_map[ys, xs])
    mean = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    mean_c = np.divide(cacc, cnt, out=np.zeros_like(cacc), where=cnt > 0)
    fallback = float(np.median(depth_map[ys, xs]))
    mean[cnt == 0] = fallback
    dense = np.repeat(np.repeat(mean, cell, axis=0), cell, axis=1)[:h, :w]
    dense_c = np.repeat(np.repeat(mean_c, cell, axis=0), cell, axis=1)[:h, :w]
    r = 2
    smooth = box_sum(dense, r) / (2 * r + 1) ** 2
    return smooth.astype(np.float32), (dense_c * 0.65).astype(np.float32)


def sfm_proposal(frame, cam, cfg, dmax: int = 64) -> DepthProposal:
    """Sparse SfM-style depth via triangulation + reprojection-error confidence.

    Uses Harris corners, epipolar scanline matching, the essential matrix for
    a geometric sanity check, batch DLT triangulation and a reprojection error
    per point. The sparse cloud is diffused into a coarse dense prior so it can
    participate in multi-source fusion.
    """
    gray_l, gray_r = frame.gray_left, frame.gray_right
    kps = harris_corners(gray_l, num=int(cfg.num_keypoints), nms_radius=3)
    disp, peak, uv = _match_keypoints(
        gray_l, gray_r, kps, int(dmax), max(1, int(cfg.match_window) // 2)
    )
    if len(uv) == 0:
        h, w = gray_l.shape
        return DepthProposal("sfm", np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), bool))

    P1, P2 = cam.projection_left(), cam.projection_right()
    x1 = uv
    x2 = np.column_stack([uv[:, 0] - disp, uv[:, 1]])
    X = triangulate_dlt(P1, P2, x1, x2)
    e1 = reprojection_error(P1, X, x1)
    e2 = reprojection_error(P2, X, x2)
    err = np.maximum(e1, e2)

    # Epipolar sanity check with the known relative pose (R=I, t=-B,0,0).
    E = essential_matrix(np.eye(3), np.array([-cam.baseline, 0.0, 0.0]))
    n1 = np.column_stack([(x1[:, 0] - cam.cx) / cam.fx, (x1[:, 1] - cam.cy) / cam.fy])
    n2 = np.column_stack([(x2[:, 0] - cam.cx) / cam.fx, (x2[:, 1] - cam.cy) / cam.fy])
    epi = epipolar_residual(n1, n2, E)

    good = (
        np.isfinite(X).all(axis=1)
        & (X[:, 2] > 0.3)
        & (X[:, 2] < 80.0)
        & (disp > 0.5)
        & (err < 6.0)
        & (epi < 0.05)
    )
    conf_pts = np.exp(-err / float(cfg.reproj_threshold)) * np.clip(peak, 0, 1)
    conf_pts = np.nan_to_num(np.clip(conf_pts, 0, 1), nan=0.0, posinf=0.0, neginf=0.0) * good

    h, w = gray_l.shape
    sparse_d = np.zeros((h, w), np.float32)
    sparse_c = np.zeros((h, w), np.float32)
    ix = np.clip(np.rint(x1[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.rint(x1[:, 1]).astype(int), 0, h - 1)
    sparse_d[iy, ix] = X[:, 2].astype(np.float32)
    sparse_c[iy, ix] = conf_pts.astype(np.float32)
    dense, dense_c = _scatter_grid(sparse_d, sparse_c, h, w)
    valid = dense_c >= 0.08
    return DepthProposal("sfm", dense, dense_c, valid)
