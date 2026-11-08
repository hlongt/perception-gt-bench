import numpy as np

from helpers import small_cfg

from gtbench.pseudo_gt import fuse
from gtbench.pseudo_gt.base import DepthProposal


def make(name, value, conf):
    shape = (8, 8)
    return DepthProposal(
        name,
        np.full(shape, value, np.float32),
        np.full(shape, conf, np.float32),
        np.ones(shape, dtype=bool),
    )


def test_fusion_downweights_outlier():
    cfg = small_cfg().fusion
    out = fuse([make("a", 5.0, 0.9), make("b", 5.0, 0.9), make("c", 50.0, 0.1)], cfg)
    assert abs(float(out.depth.mean()) - 5.0) < 0.6


def test_single_source_is_passthrough():
    cfg = small_cfg().fusion
    a = make("a", 7.0, 0.8)
    out = fuse([a], cfg)
    assert np.allclose(out.depth, 7.0)
    assert np.all(out.valid)


def test_confidence_monotone_with_agreement():
    cfg = small_cfg().fusion
    agree = fuse([make("a", 5.0, 0.8), make("b", 5.0, 0.8)], cfg)
    disagree = fuse([make("a", 5.0, 0.8), make("b", 9.0, 0.8)], cfg)
    assert agree.confidence.mean() >= disagree.confidence.mean()
