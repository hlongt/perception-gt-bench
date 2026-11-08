from __future__ import annotations

import warnings

import numpy as np

from .base import DepthProposal


def fuse(proposals: list[DepthProposal], cfg, name: str = "fused",
         source_weights: dict | None = None) -> DepthProposal:
    """Confidence-weighted fusion with a MAD-based outlier gate.

    Instead of a naive average, each pixel first compares every source against
    the per-pixel median. Sources whose deviation exceeds k * MAD are down-
    weighted rather than dropped, so a single confident-but-wrong source
    cannot dominate while a temporarily weak source can still contribute.

    ``source_weights`` applies per-source reliability estimated on a small
    held-out calibration set. Without it, sources with incomparable confidence
    scales (a dense stereo match vs. a relative monocular prior) will drag the
    fused depth towards the least accurate source -- see the naive ablation in
    the report.
    """
    if not proposals:
        raise ValueError("no proposals to fuse")
    n = len(proposals)
    stack = np.stack([p.depth for p in proposals]).astype(np.float32)
    raw_conf = np.stack([p.confidence * p.valid for p in proposals]).astype(np.float32)
    if n == 1:
        return DepthProposal(name, stack[0].copy(), raw_conf[0].copy(), proposals[0].valid.copy())

    if source_weights:
        scale = np.array(
            [float(source_weights.get(p.name, 1.0)) for p in proposals], dtype=np.float32
        )[:, None, None]
        conf = raw_conf * scale
    else:
        conf = raw_conf.copy()

    masked = np.where(conf > 1e-6, stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nan_to_num(np.nanmedian(masked, axis=0), nan=0.0)
        dev = np.abs(stack - med[None])
        mad = np.nan_to_num(
            np.nanmedian(np.where(conf > 1e-6, dev, np.nan), axis=0), nan=0.0
        )
    thresh = float(cfg.mad_k) * np.maximum(mad, 0.05)
    outlier = dev > thresh[None]
    conf = np.where(outlier, conf * 0.05, conf)

    wsum = conf.sum(axis=0)
    depth = np.divide(
        (stack * conf).sum(axis=0),
        wsum,
        out=np.zeros_like(stack[0]),
        where=wsum > 1e-6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spread = np.nan_to_num(
            np.nanstd(np.where(conf > 1e-6, stack, np.nan), axis=0), nan=0.0
        )
    agreement = np.clip(1.0 - spread / (np.maximum(depth, 0.5) + 1e-6), 0.0, 1.0)
    raw_wsum = raw_conf.sum(axis=0)
    confidence = np.clip((raw_wsum / n) * (0.55 + 0.45 * agreement), 0.0, 1.0)
    valid = raw_wsum > float(cfg.min_confidence)
    return DepthProposal(name, depth.astype(np.float32), confidence.astype(np.float32), valid)
