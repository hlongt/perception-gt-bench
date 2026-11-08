from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np

from ..data import build_dataset
from ..gt_qc import aggregate_quality, quality_report
from ..native import status
from ..pseudo_gt import foundation_proposal, fuse, sfm_proposal, stereo_proposal
from .metrics import depth_metrics, tracking_metrics
from .slices import group_by_slice, rank_slices
from .tracker import IoU3DTracker

VARIANTS = OrderedDict(
    [
        ("stereo_only", ["stereo"]),
        ("sfm_only", ["sfm"]),
        ("offline_prior_only", ["offline_prior"]),
        ("stereo_plus_sfm", ["stereo", "sfm"]),
        ("fused_naive", ["stereo", "sfm", "offline_prior"]),
        ("fused_calibrated", ["stereo", "sfm", "offline_prior"]),
    ]
)

VARIANT_LABELS = {
    "stereo_only": "仅双目立体匹配",
    "sfm_only": "仅 SfM 稀疏重建",
    "offline_prior_only": "仅离线单目先验",
    "stereo_plus_sfm": "双目 + SfM 等权融合",
    "fused_naive": "三源等权融合（未校准，对照组）",
    "fused_calibrated": "三源 + 源级可靠性校准（最终方案）",
}

FINAL_VARIANT = "fused_calibrated"
NAIVE_VARIANT = "fused_naive"


def build_proposals(frame, cam, cfg) -> dict:
    s = stereo_proposal(frame, cam, cfg.stereo)
    f = sfm_proposal(frame, cam, cfg.sfm, dmax=int(cfg.stereo.max_disparity))
    p = foundation_proposal(frame, cam, cfg.fusion, sfm_ref=f)
    return {"stereo": s, "sfm": f, "offline_prior": p}


def calibrate_source_weights(cfg, cam) -> tuple[dict, dict]:
    """Estimate per-source reliability on a held-out calibration set.

    A real GT system has a small LiDAR / manual-verification calibration set.
    We emulate it with frames generated from a different seed, so no评测帧
    leaks into calibration. Reliability is inverse squared error, normalised
    so the best source has weight 1.0.
    """
    calib = build_dataset(cfg, int(cfg.seed) + 777)
    groups = group_by_slice(calib["frames"])
    frames = [g[0] for g in groups.values()]
    acc = defaultdict(list)
    for f in frames:
        for name, proposal in build_proposals(f, cam, cfg).items():
            rep = quality_report(proposal, f)
            if rep.get("absrel") is not None:
                acc[name].append(rep["absrel"])
    errors = {n: round(float(np.mean(v)), 5) for n, v in acc.items()}
    raw = {n: 1.0 / (0.05 + e) ** 2 for n, e in errors.items()}
    peak = max(raw.values()) if raw else 1.0
    weights = {n: round(v / peak, 5) for n, v in raw.items()}
    info = {
        "calibration_frames": len(frames),
        "calibration_seed": int(cfg.seed) + 777,
        "source_absrel": errors,
        "reliability_weights": weights,
        "note": "独立 seed 的留出标定集，评测帧不参与标定；逆误差平方定权并归一化",
    }
    return weights, info


def _tile_badcases(frame, proposal, k: int = 10, tile: int = 16) -> list[dict]:
    gt = frame.depth_gt
    pred = proposal.depth
    valid = proposal.valid & frame.valid_gt & (gt > 0.3) & (pred > 0.05)
    err = np.where(valid, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
    h, w = err.shape
    out = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            t = err[y : y + tile, x : x + tile]
            if np.isfinite(t).sum() < t.size * 0.5:
                continue
            out.append(
                {"x": int(x), "y": int(y),
                 "absrel": round(float(np.nanmean(t)), 4),
                 "coverage": round(float(np.isfinite(t).mean()), 3)}
            )
    out.sort(key=lambda d: -d["absrel"])
    return out[:k]


def evaluate(dataset, cfg, manifest=None, verbose: bool = True) -> dict:
    cam = dataset["camera"]
    frames = dataset["frames"]
    records: list[dict] = []
    badcases: list[dict] = []
    per_slice = defaultdict(lambda: defaultdict(list))

    if verbose:
        print("  calibrating per-source reliability on held-out frames ...", flush=True)
    weights, calib_info = calibrate_source_weights(cfg, cam)
    common_acc = defaultdict(lambda: [0.0, 0])
    risk_acc = defaultdict(list)
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for i, frame in enumerate(frames):
        props = build_proposals(frame, cam, cfg)
        frame_proposals = {}
        for vname, sources in VARIANTS.items():
            proposal = (
                props[sources[0]]
                if len(sources) == 1
                else fuse(
                    [props[s] for s in sources],
                    cfg.fusion,
                    name=vname,
                    source_weights=weights if vname == FINAL_VARIANT else None,
                )
            )
            rep = quality_report(proposal, frame)
            rep["variant"] = vname
            records.append(rep)
            frame_proposals[vname] = proposal
            for key in ("absrel", "rmse", "silog", "delta_1_25", "coverage"):
                if rep.get(key) is not None:
                    per_slice[vname][frame.slice_name].append((key, rep[key]))
            if vname == FINAL_VARIANT:
                for bc in _tile_badcases(frame, proposal, k=int(cfg.benchmark.topk_badcases)):
                    badcases.append(
                        {"slice": frame.slice_name, "frame_idx": frame.frame_idx, **bc}
                    )

        # Fair comparison: evaluate every variant on the SAME pixel set where
        # all variants are valid. Otherwise a sparse high-accuracy source is
        # compared against a dense source on different pixels.
        common = frame.valid_gt & (frame.depth_gt > 0.3)
        for p in frame_proposals.values():
            common &= p.valid
        gt = frame.depth_gt
        if int(common.sum()) > 100:
            for vname, p in frame_proposals.items():
                m = common & (p.depth > 0.05)
                e = np.abs(p.depth[m] - gt[m]) / np.maximum(gt[m], 1e-6)
                common_acc[vname][0] += float(e.sum())
                common_acc[vname][1] += int(len(e))

        # Risk-coverage curve: how much accuracy do we trade for coverage by
        # accepting lower-confidence fused pseudo-labels?
        fp = frame_proposals[FINAL_VARIANT]
        base = frame.valid_gt & (gt > 0.3) & (fp.depth > 0.05)
        for t in thresholds:
            m = base & (fp.confidence >= t)
            if int(m.sum()) > 100:
                risk_acc[t].append(
                    (float(m.mean()),
                     float((np.abs(fp.depth[m] - gt[m]) / np.maximum(gt[m], 1e-6)).mean()))
                )

        if manifest is not None:
            manifest.mark(frame, variants=len(VARIANTS), status="evaluated")
        if verbose and (i + 1) % 6 == 0:
            print(f"  evaluated {i+1}/{len(frames)} frames", flush=True)
    if manifest is not None:
        manifest.save()

    by_variant = {}
    for vname in VARIANTS:
        rs = [r for r in records if r["variant"] == vname]
        agg = aggregate_quality(rs)
        agg["label"] = VARIANT_LABELS[vname]
        by_variant[vname] = agg

    per_slice_out = {}
    for vname, slices in per_slice.items():
        per_slice_out[vname] = {
            s: {k: round(float(np.mean([x for kk, x in vals if kk == k])), 5)
                for k in ("absrel", "rmse", "silog", "delta_1_25", "coverage")}
            for s, vals in slices.items()
        }

    tracker = IoU3DTracker(float(cfg.benchmark.iou_threshold), max_age=2)
    seq = []
    for f in dataset["track_sequence"]:
        gt = [{"id": o["id"], "box3d": o["box3d"]} for o in f.objects_gt]
        dets = [{"id": i, "box3d": d["box3d"]} for i, d in enumerate(f.detections)]
        seq.append((gt, tracker.update(dets)))
    tracking = tracking_metrics(seq, float(cfg.benchmark.iou_threshold))

    badcases.sort(key=lambda d: -d["absrel"])
    fused = [r for r in records if r["variant"] == FINAL_VARIANT]
    common_domain = {v: round(s / max(n, 1), 5) for v, (s, n) in common_acc.items()}
    risk_coverage = [
        {
            "confidence_threshold": t,
            "coverage": round(float(np.mean([x[0] for x in vals])), 4),
            "absrel": round(float(np.mean([x[1] for x in vals])), 5),
        }
        for t, vals in sorted(risk_acc.items())
        if vals
    ]
    return {
        "meta": {
            "frames": len(frames),
            "slices": list(cfg.dataset.slices),
            "track_frames": len(seq),
            "variants": list(VARIANTS),
            "final_variant": FINAL_VARIANT,
            "naive_variant": NAIVE_VARIANT,
            "resolution": [int(cfg.camera.width), int(cfg.camera.height)],
            "seed": int(cfg.seed),
        },
        "variants": by_variant,
        "per_slice": per_slice_out,
        "slice_ranking": rank_slices(fused),
        "tracking": tracking,
        "badcases": badcases[: int(cfg.benchmark.topk_badcases)],
        "records": records,
        "native": status(),
        "common_domain": common_domain,
        "risk_coverage": risk_coverage,
        "source_calibration": calib_info,
        "calibration": next(
            (r["confidence_calibration"] for r in fused if r["confidence_calibration"]), []
        ),
    }
