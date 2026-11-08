from __future__ import annotations

import argparse
from pathlib import Path

from .bench_native import benchmark as native_benchmark
from .benchmark.regression import build_proposals, evaluate
from .benchmark.report import write_html, write_json, write_markdown
from .benchmark.viz import generate_all
from .config import REPO_ROOT, load_config
from .data import build_dataset
from .data.ingest import Manifest
from .native import status
from .pseudo_gt import fuse


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="PerceptionGT-Bench one-command pipeline")
    ap.add_argument("--config", default=None, help="path to a JSON config")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="override dataset seed")
    ap.add_argument("--no-viz", action="store_true", help="skip figure generation")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else REPO_ROOT / cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("PerceptionGT-Bench | pseudo-GT production + benchmark")
    print(f"  config={args.config or 'configs/default.json'}  out={out}")
    print(f"  native backend: {status()}")

    print("[1/5] building deterministic stereo dataset ...")
    ds = build_dataset(cfg, args.seed)
    manifest = Manifest(out / "manifest.json")
    new = sum(1 for f in ds["frames"] if not manifest.seen(f))
    print(f"  frames={len(ds['frames'])}  new={new}  (sha256 manifest enables resume)")

    print("[2/5] producing pseudo-GT and evaluating 6 variants ...")
    results = evaluate(ds, cfg, manifest=manifest)

    print("[3/5] benchmarking C++17 kernels ...")
    results["native_benchmark"] = native_benchmark()
    results["native"] = status()

    print("[4/5] writing reports ...")
    write_json(results, out / "results.json")
    write_markdown(results, out / "report.md")
    write_html(results, out / "report.html")

    if not args.no_viz:
        print("[5/5] rendering figures ...")
        frame = ds["frames"][0]
        props = build_proposals(frame, ds["camera"], cfg)
        proposal = fuse(
            [props[s] for s in ("stereo", "sfm", "offline_prior")],
            cfg.fusion,
            name=results["meta"]["final_variant"],
            source_weights=results["source_calibration"]["reliability_weights"],
        )
        print("  " + ", ".join(generate_all(results, frame, proposal, out)))
    else:
        print("[5/5] figures skipped")

    print("\n=== variant comparison (AbsRel, lower is better) ===")
    for name, x in results["variants"].items():
        print(f"  {name:22s} AbsRel={x.get('absrel')}  d1={x.get('delta_1_25')}  cov={x.get('coverage')}")
    t = results["tracking"]
    print(f"\n=== 3D tracking ===\n  MOTA={t['MOTA']}  IDF1={t['IDF1']}  IDSW={t['IDSW']}  HOTA*={t['HOTA_single_threshold']}")
    nb = results["native_benchmark"]
    print(f"\n=== C++ kernels ===\n  loaded={nb['native_loaded']}  median3 speedup={nb['median3']['speedup']}x  combo speedup={nb['combo']['speedup']}x")
    print(f"\nReports: {out / 'report.md'} | {out / 'report.html'} | {out / 'results.json'}")
    return results


if __name__ == "__main__":
    main()
