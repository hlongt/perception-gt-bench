"""Pseudo-ground-truth production: SfM, stereo and foundation-model backends."""

from .base import DepthProposal
from .fusion import fuse
from .foundation import DepthAnythingV2Backend, GradientPriorBackend, foundation_proposal
from .sfm import sfm_proposal
from .stereo import stereo_proposal

__all__ = [
    "DepthProposal",
    "DepthAnythingV2Backend",
    "GradientPriorBackend",
    "foundation_proposal",
    "sfm_proposal",
    "stereo_proposal",
    "fuse",
]
