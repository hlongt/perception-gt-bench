from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..geometry import box_sum, fit_scale_shift
from .base import DepthProposal


class FoundationBackend(ABC):
    """Interface for an offline foundation model producing relative depth."""

    name = "backend"

    @abstractmethod
    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (relative_depth, confidence), both (H, W) float32."""


class GradientPriorBackend(FoundationBackend):
    """Hand-crafted, dependency-free monocular prior.

    This is an OFFLINE LIGHTWEIGHT STAND-IN, not a learned model. It combines
    a vertical-position prior, a luma cue and edge-aware smoothing. It exists
    so the fusion / benchmark chain is fully reproducible without downloading
    multi-hundred-MB checkpoints. It is deliberately low-confidence.
    """

    name = "offline_prior"

    def predict(self, rgb):
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        h, w = g.shape
        vv = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        prior = 1.0 - 0.42 * vv - 0.16 * g
        prior = box_sum(prior.astype(np.float64), 3) / 49.0
        gx = np.zeros_like(g)
        gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
        conf = np.clip(0.30 - np.clip(gx, 0, 1) * 0.18, 0.05, 0.32)
        return prior.astype(np.float32), conf.astype(np.float32)


class DepthAnythingV2Backend(FoundationBackend):
    """Integration point for a real monocular foundation model.

    NOT EXECUTED BY DEFAULT: it needs torch and a downloaded checkpoint. The
    repo never silently pretends a large model ran. See docs/复现指南.md for
    the exact wiring (swap this backend into foundation_proposal).
    """

    name = "depth_anything_v2"

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint

    def predict(self, rgb):
        raise RuntimeError(
            "DepthAnythingV2Backend requires torch + a local checkpoint. "
            "Pass backend=GradientPriorBackend() for the offline reproducible run."
        )


class SAMBackend(FoundationBackend):
    """Integration point for semantic pseudo-labels (same contract)."""

    name = "sam"

    def predict(self, rgb):
        raise RuntimeError("SAMBackend requires torch + a local checkpoint.")


def foundation_proposal(frame, cam, cfg, sfm_ref=None, backend=None) -> DepthProposal:
    """Relative depth -> metric depth via least-squares alignment to SfM."""
    backend = backend or GradientPriorBackend()
    rel, conf = backend.predict(frame.rgb_left)
    if sfm_ref is not None:
        a, b = fit_scale_shift(
            rel, sfm_ref.depth, sfm_ref.valid & (sfm_ref.confidence > 0.2)
        )
    else:
        a, b = 4.0, 0.5
    depth = np.clip(a * rel + b, 0.2, 80.0).astype(np.float32)
    valid = (conf > 0.05) & (depth > 0.2)
    return DepthProposal(backend.name, depth, conf, valid)
