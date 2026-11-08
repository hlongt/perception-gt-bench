from __future__ import annotations

import numpy as np

from .benchmark.metrics import depth_metrics


def quality_report(proposal, frame, bins: int = 5) -> dict:
    """GT quality control plus confidence calibration for one pseudo-GT source."""
    valid = (
        proposal.valid
        & frame.valid_gt
        & (frame.depth_gt > 0.3)
        & (proposal.depth > 0.05)
    )
    out = depth_metrics(proposal.depth, frame.depth_gt, valid)
    out["source"] = proposal.name
    out["slice"] = frame.slice_name
    out["frame_idx"] = frame.frame_idx
    out["coverage"] = round(float(valid.mean()), 4)

    err = np.abs(proposal.depth.astype(np.float64) - frame.depth_gt) / np.maximum(
        frame.depth_gt, 1e-6
    )
    calibration = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        sel = valid & (proposal.confidence >= edges[i])
        if i < bins - 1:
            sel &= proposal.confidence < edges[i + 1]
        if sel.sum() >= 20:
            calibration.append(
                {
                    "conf_bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                    "pixels": int(sel.sum()),
                    "mean_absrel": round(float(err[sel].mean()), 4),
                }
            )
    out["confidence_calibration"] = calibration
    return out


def aggregate_quality(reports: list[dict]) -> dict:
    keys = ["absrel", "rmse", "silog", "delta_1_25", "coverage"]
    out = {}
    for k in keys:
        vals = [r[k] for r in reports if k in r and np.isfinite(r[k])]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out
