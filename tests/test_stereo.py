import numpy as np

from helpers import small_cfg

from gtbench.benchmark.regression import build_proposals
from gtbench.data import build_dataset


def test_stereo_recovers_depth_ordering():
    cfg = small_cfg()
    ds = build_dataset(cfg)
    f = ds["frames"][0]  # normal slice
    p = build_proposals(f, ds["camera"], cfg)["stereo"]
    assert p.valid.mean() > 0.15, "stereo matching should produce a usable mask"
    gt = f.depth_gt
    near = (gt > 1.5) & (gt < 2.8) & p.valid
    far = (gt > 7.0) & (gt < 8.5) & p.valid
    assert near.sum() > 50 and far.sum() > 50
    assert p.depth[near].mean() < p.depth[far].mean()


def test_subpixel_disparity_is_not_integer_only():
    from gtbench.pseudo_gt.stereo import compute_disparity

    cfg = small_cfg()
    ds = build_dataset(cfg)
    f = ds["frames"][0]
    disp, _, _ = compute_disparity(f.gray_left, f.gray_right, cfg.stereo)
    frac = np.abs(disp - np.rint(disp))
    assert np.mean(frac > 1e-4) > 0.1, "parabola fit should yield sub-pixel values"
