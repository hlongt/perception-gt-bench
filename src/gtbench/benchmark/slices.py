from __future__ import annotations

from collections import defaultdict

SLICE_DESCRIPTIONS = {
    "normal": "正常纹理与光照，作为基准场景",
    "low_texture": "弱纹理 / 重复纹理，考察匹配歧义",
    "dynamic": "帧内运动与双快门不一致，考察动态物体",
    "low_light": "低照度 + 传感器噪声，考察退化条件下的鲁棒性",
    "motion_blur": "水平运动模糊，考察边缘定位精度",
    "far": "远距离小视差，考察视差量化误差",
}


def group_by_slice(frames) -> dict:
    out = defaultdict(list)
    for f in frames:
        out[f.slice_name].append(f)
    return dict(out)


def rank_slices(reports, key: str = "absrel") -> list[dict]:
    by = defaultdict(list)
    for r in reports:
        v = r.get(key)
        if v is not None and r.get("n", 0) > 0:
            by[r["slice"]].append(v)
    ranked = [
        {"slice": s, key: round(float(sum(v) / len(v)), 5), "frames": len(v),
         "description": SLICE_DESCRIPTIONS.get(s, "")}
        for s, v in by.items()
    ]
    return sorted(ranked, key=lambda x: -x[key])
