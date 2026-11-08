import numpy as np

from helpers import small_cfg

from gtbench.benchmark.regression import build_proposals, evaluate
from gtbench.data import build_dataset
from gtbench.data.ingest import Manifest, frame_digest


def test_dataset_is_deterministic():
    cfg = small_cfg()
    a = build_dataset(cfg)["frames"][0]
    b = build_dataset(cfg)["frames"][0]
    assert np.array_equal(a.rgb_left, b.rgb_left)
    assert np.array_equal(a.depth_gt, b.depth_gt)


def test_manifest_dedup_resume(tmp_path):
    cfg = small_cfg()
    f = build_dataset(cfg)["frames"][0]
    m = Manifest(tmp_path / "manifest.json")
    assert not m.seen(f)
    m.mark(f, status="evaluated")
    m.save()
    m2 = Manifest(tmp_path / "manifest.json")
    assert m2.seen(f)
    assert frame_digest(f) == m2.records[f"{f.slice_name}/{f.frame_idx:04d}"]["digest"]


def test_proposals_shape_and_sources():
    cfg = small_cfg()
    ds = build_dataset(cfg)
    f = ds["frames"][0]
    p = build_proposals(f, ds["camera"], cfg)
    assert set(p) == {"stereo", "sfm", "offline_prior"}
    for x in p.values():
        assert x.depth.shape == f.depth_gt.shape
        assert x.confidence.min() >= 0.0 and x.confidence.max() <= 1.0


def test_evaluate_smoke():
    cfg = small_cfg()
    ds = build_dataset(cfg)
    r = evaluate(ds, cfg, verbose=False)
    assert len(r["records"]) == 2 * 6
    assert set(r["variants"]) == {
        "stereo_only", "sfm_only", "offline_prior_only", "stereo_plus_sfm",
        "fused_naive", "fused_calibrated"
    }
    assert "MOTA" in r["tracking"]
    assert len(r["slice_ranking"]) == 2
    assert r["meta"]["frames"] == 2
    assert r["meta"]["final_variant"] == "fused_calibrated"
    assert r["variants"]["fused_calibrated"]["absrel"] < r["variants"]["fused_naive"]["absrel"]
