"""Dataset layer: deterministic synthetic stereo renderer and ingest manifest."""

from .calibration import Camera
from .synth import Frame, build_dataset

__all__ = ["Camera", "Frame", "build_dataset"]
