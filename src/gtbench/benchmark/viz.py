from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_depth_panel(frame, proposal, path):
    gt = frame.depth_gt.astype(np.float64)
    pred = proposal.depth.astype(np.float64)
    valid = proposal.valid & frame.valid_gt & (gt > 0.3) & (pred > 0.05)
    err = np.where(valid, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
    hi = float(np.nanpercentile(pred[valid], 98)) if valid.any() else 1.0
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.6))
    ax[0, 0].imshow(frame.rgb_left)
    ax[0, 0].set_title("Rectified left RGB")
    im = ax[0, 1].imshow(np.where(frame.valid_gt, gt, np.nan), cmap="magma", vmin=0, vmax=hi)
    ax[0, 1].set_title("Depth ground truth (m)")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    im = ax[1, 0].imshow(np.where(proposal.valid, pred, np.nan), cmap="magma", vmin=0, vmax=hi)
    ax[1, 0].set_title("Fused pseudo-GT (m)")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
    emax = float(np.nanpercentile(err, 95)) if np.isfinite(err).any() else 0.5
    im = ax[1, 1].imshow(err, cmap="inferno", vmin=0, vmax=max(emax, 0.01))
    ax[1, 1].set_title("AbsRel error map")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_slice_bars(results, path):
    fv = results["meta"]["final_variant"]
    data = results["per_slice"].get(fv, {})
    names = list(data)
    vals = [data[s].get("absrel") or 0 for s in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, vals, color="#BA7517")
    ax.set_ylabel("AbsRel (lower is better)")
    ax.set_title("Difficulty by scene slice - fused pseudo-GT")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_variant_bars(results, path):
    short = {
        "stereo_only": "stereo",
        "sfm_only": "SfM",
        "offline_prior_only": "offline prior",
        "stereo_plus_sfm": "stereo+SfM",
        "fused_naive": "fused naive",
        "fused_calibrated": "fused calibrated",
    }
    v = results["variants"]
    keys = list(v)
    names = [short.get(k, k) for k in keys]
    vals = [v[k].get("absrel") or 0 for k in keys]
    colors = ["#888780"] * len(vals)
    fv = results["meta"]["final_variant"]
    if fv in keys:
        colors[keys.index(fv)] = "#185FA5"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("AbsRel (lower is better)")
    ax.set_title("Pseudo-GT source / fusion ablation")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_risk_coverage(results, path):
    rc = results.get("risk_coverage", [])
    if not rc:
        return
    x = [r["confidence_threshold"] for r in rc]
    cov = [r["coverage"] for r in rc]
    err = [r["absrel"] for r in rc]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(x, cov, "o-", color="#185FA5", label="coverage")
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("coverage", color="#185FA5")
    ax1.tick_params(axis="y", labelcolor="#185FA5")
    ax2 = ax1.twinx()
    ax2.plot(x, err, "s-", color="#BA7517", label="AbsRel")
    ax2.set_ylabel("AbsRel", color="#BA7517")
    ax2.tick_params(axis="y", labelcolor="#BA7517")
    ax1.grid(alpha=0.3)
    ax1.set_title("Risk-coverage trade-off of fused pseudo-GT")
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def generate_all(results, frame, proposal, outdir) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, name in (
        (lambda p: plot_depth_panel(frame, proposal, p), "depth_panel.png"),
        (lambda p: plot_slice_bars(results, p), "slice_bars.png"),
        (lambda p: plot_variant_bars(results, p), "variant_bars.png"),
        (lambda p: plot_risk_coverage(results, p), "risk_coverage.png"),
    ):
        target = outdir / name
        fn(target)
        made.append(name)
    return made
