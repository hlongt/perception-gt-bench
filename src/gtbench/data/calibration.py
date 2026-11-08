from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Rectified pinhole stereo camera. Left camera is the world origin.

    The optical axis is +Z, +Y points down, +X points right. A point at
    depth Z projects to disparity d = fx * baseline / Z.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    @classmethod
    def from_config(cls, c) -> "Camera":
        return cls(
            width=int(c.width),
            height=int(c.height),
            fx=float(c.fx),
            fy=float(c.fy),
            cx=float(c.cx),
            cy=float(c.cy),
            baseline=float(c.baseline),
        )

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def ray_directions(self) -> np.ndarray:
        """(H, W, 3) unit-Z ray directions in camera frame."""
        u = np.arange(self.width, dtype=np.float64) + 0.5
        v = np.arange(self.height, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(u, v)
        return np.stack(
            [(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, np.ones_like(uu)],
            axis=-1,
        )

    def disparity_from_depth(self, z: np.ndarray) -> np.ndarray:
        return self.fx * self.baseline / np.maximum(z, 1e-6)

    def depth_from_disparity(self, d: np.ndarray) -> np.ndarray:
        return self.fx * self.baseline / np.maximum(d, 1e-6)

    def backproject(self, u, v, z) -> np.ndarray:
        return np.stack(
            [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=-1
        )

    def project(self, X: np.ndarray) -> np.ndarray:
        z = np.maximum(X[..., 2], 1e-6)
        return np.stack(
            [X[..., 0] * self.fx / z + self.cx, X[..., 1] * self.fy / z + self.cy],
            axis=-1,
        )

    def projection_left(self) -> np.ndarray:
        """P1 = K [I | 0], shape (3, 4)."""
        return np.hstack([self.K, np.zeros((3, 1))])

    def projection_right(self) -> np.ndarray:
        """P2 = K [I | t] with t = (-baseline, 0, 0).

        The right camera is translated by +baseline in X, so points expressed
        in the right camera frame are X_left - t_right.
        """
        t = np.array([[-self.baseline], [0.0], [0.0]])
        return np.hstack([self.K, self.K @ t])
