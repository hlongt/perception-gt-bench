from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthProposal:
    """One depth estimate from one pseudo-GT source.

    depth:       (H, W) float32 metric depth in metres; 0 means invalid.
    confidence: (H, W) float32 in [0, 1], source-specific reliability.
    valid:      (H, W) bool mask of pixels the source is willing to assert.
    """

    name: str
    depth: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.confidence = np.clip(np.asarray(self.confidence, dtype=np.float32), 0.0, 1.0)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.depth[~self.valid] = 0.0
        self.confidence[~self.valid] = 0.0
