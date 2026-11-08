from __future__ import annotations

import numpy as np


def skew(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]
    )


def essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E = [t]_x R; satisfies x2^T E x1 = 0 for normalized correspondences."""
    return skew(t) @ R


def epipolar_residual(x1: np.ndarray, x2: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Sampson-like algebraic residual for (N, 2) normalized coordinates."""
    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])
    ex1 = h1 @ E.T
    ex2 = h2 @ E
    denom = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + ex2[:, 0] ** 2 + ex2[:, 1] ** 2
    return np.abs(np.einsum("ij,ij->i", h2, ex1)) / np.sqrt(np.maximum(denom, 1e-12))


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Batch DLT triangulation.

    Args:
        P1, P2: (3, 4) projection matrices.
        x1, x2: (N, 2) corresponding pixel coordinates.
    Returns:
        (N, 3) Euclidean 3D points.
    """
    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    n = len(x1)
    A = np.empty((n, 4, 4), dtype=np.float64)
    A[:, 0, :] = x1[:, 0:1] * P1[2] - P1[0]
    A[:, 1, :] = x1[:, 1:2] * P1[2] - P1[1]
    A[:, 2, :] = x2[:, 0:1] * P2[2] - P2[0]
    A[:, 3, :] = x2[:, 1:2] * P2[2] - P2[1]
    _, _, vh = np.linalg.svd(A)
    Xh = vh[:, -1, :]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:4]) < 1e-12, 1e-12, Xh[:, 3:4])
    return X


def reprojection_error(P: np.ndarray, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    Xh = np.column_stack([X, np.ones(len(X))])
    p = Xh @ P.T
    q = p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-12, 1e-12, p[:, 2:3])
    return np.linalg.norm(q - x, axis=1)


def fit_scale_shift(rel: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple:
    """Least-squares alignment ref ~= a * rel + b. Returns (a, b)."""
    m = mask & np.isfinite(rel) & np.isfinite(ref) & (ref > 0)
    if m.sum() < 8:
        return 1.0, 0.0
    r = rel[m]
    y = ref[m]
    A = np.column_stack([r, np.ones_like(r)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def box_sum(img: np.ndarray, radius: int) -> np.ndarray:
    """Integral-image box filter, used for SAD and the structure tensor."""
    a = np.asarray(img, dtype=np.float64)
    p = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = a.shape
    k = 2 * radius + 1
    return (
        ii[k : k + h, k : k + w]
        - ii[:h, k : k + w]
        - ii[k : k + h, :w]
        + ii[:h, :w]
    )


def harris_corners(gray: np.ndarray, num: int = 300, nms_radius: int = 3, k: float = 0.04):
    """Corner detection without OpenCV: gradients -> structure tensor -> NMS."""
    g = np.asarray(gray, dtype=np.float64)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    r = 1
    sxx = box_sum(gx * gx, r)
    syy = box_sum(gy * gy, r)
    sxy = box_sum(gx * gy, r)
    det = sxx * syy - sxy**2
    tr = sxx + syy
    resp = det - k * tr**2
    resp[:nms_radius, :] = -1
    resp[-nms_radius:, :] = -1
    resp[:, :nms_radius] = -1
    resp[:, -nms_radius:] = -1
    flat = np.argsort(resp.ravel())[::-1][: num * 12]
    chosen = []
    h, w = resp.shape
    for idx in flat:
        v, u = divmod(int(idx), w)
        if resp[v, u] <= 1e-9:
            break
        if any(abs(v - pv) <= nms_radius and abs(u - pu) <= nms_radius for pv, pu in chosen):
            continue
        chosen.append((v, u))
        if len(chosen) >= num:
            break
    return np.array([[u, v] for v, u in chosen], dtype=np.float64).reshape(-1, 2)
