from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import Camera


@dataclass
class Frame:
    slice_name: str
    frame_idx: int
    rgb_left: np.ndarray
    rgb_right: np.ndarray
    depth_gt: np.ndarray
    valid_gt: np.ndarray
    objects_gt: list[dict[str, Any]] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.rgb_left.shape[0]

    @property
    def width(self) -> int:
        return self.rgb_left.shape[1]

    @property
    def gray_left(self) -> np.ndarray:
        return _gray(self.rgb_left)

    @property
    def gray_right(self) -> np.ndarray:
        return _gray(self.rgb_right)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
        np.float32
    )


_BASE_SPECS = [
    (2.2, -0.95, -0.20, 0.42, (196, 118, 92)),
    (3.1, 0.75, -0.10, 0.50, (102, 152, 196)),
    (4.4, -0.35, 0.10, 0.60, (128, 178, 122)),
    (6.0, 1.10, -0.05, 0.72, (186, 166, 108)),
]


def _make_boxes(slice_name: str, frame_idx: int, right_view: bool):
    scale = 2.2 if slice_name == "far" else 1.0
    tex = 0.03 if slice_name == "low_texture" else 1.0
    motion = 0.16 if slice_name == "dynamic" else 0.055
    boxes = [
        {
            "lo": np.array([-40.0, -40.0, 8.0 * scale]),
            "hi": np.array([40.0, 40.0, 8.0 * scale + 0.25]),
            "color": np.array([148.0, 156.0, 168.0]),
            "oid": -1,
            "tex": tex,
        },
        {
            "lo": np.array([-40.0, 1.15, 0.35]),
            "hi": np.array([40.0, 1.35, 40.0 * scale]),
            "color": np.array([112.0, 110.0, 106.0]),
            "oid": -2,
            "tex": tex,
        },
    ]
    # A small inter-camera offset on the dynamic slice emulates intra-frame
    # motion / rolling-shutter mismatch between the two shutters.
    shutter = 0.035 if (slice_name == "dynamic" and right_view) else 0.0
    for i, (z, cx, cy, half, color) in enumerate(_BASE_SPECS):
        z *= scale
        half *= scale
        drift = motion * frame_idx * (1.0 if i % 2 == 0 else -1.0) + shutter
        lo = np.array([cx - half + drift, cy - half, z])
        hi = np.array([cx + half + drift, cy + half, z + 0.8 * scale])
        boxes.append(
            {
                "lo": lo,
                "hi": hi,
                "color": np.array(color, dtype=np.float64),
                "oid": i,
                "tex": tex,
            }
        )
    return boxes


def _render(cam: Camera, boxes, origin_x: float, slice_name: str, rng) -> tuple:
    dirs = cam.ray_directions()
    h, w = dirs.shape[:2]
    o = np.array([origin_x, 0.0, 0.0])
    safe = np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    best_t = np.full((h, w), np.inf, dtype=np.float64)
    best_id = np.full((h, w), -99, dtype=np.int32)
    hit_point = np.zeros((h, w, 3), dtype=np.float64)

    for i, box in enumerate(boxes):
        t0 = (box["lo"] - o) / safe
        t1 = (box["hi"] - o) / safe
        tn = np.minimum(t0, t1).max(axis=-1)
        tf = np.maximum(t0, t1).min(axis=-1)
        hit = (tf >= tn) & (tn > 0.05) & (tn < best_t)
        best_t = np.where(hit, tn, best_t)
        best_id = np.where(hit, i, best_id)
        hit_point = np.where(hit[..., None], o + tn[..., None] * dirs, hit_point)

    valid = np.isfinite(best_t)
    depth = np.where(valid, best_t, 0.0).astype(np.float32)

    # World-coordinate texture: the same surface point gets the same colour in
    # both views, so stereo correspondence is genuinely solvable.
    X, Y, Z = hit_point[..., 0], hit_point[..., 1], hit_point[..., 2]
    checker = np.mod(np.floor(X * 5.0) + np.floor(Y * 5.0), 2.0)
    stripes = 0.5 + 0.5 * np.sin(Z * 2.7 + X * 7.0)
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for i, box in enumerate(boxes):
        m = best_id == i
        if not m.any():
            continue
        amp = box["tex"]
        shade = 0.62 + 0.30 * checker * amp + 0.16 * stripes * amp
        rgb[m] = box["color"] * shade[m, None]
    rgb[~valid] = 12.0

    if slice_name == "motion_blur":
        pad = np.pad(rgb, ((0, 0), (1, 1), (0, 0)), mode="edge")
        rgb = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0
    if slice_name == "low_light":
        rgb = rgb * 0.42 + rng.normal(0.0, 5.0, rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, depth, valid


def _noisy_detections(boxes, rng, iou_threshold_noise: float = 0.035):
    dets = []
    for box in boxes:
        if box["oid"] < 0:
            continue
        if rng.random() < 0.06:
            continue  # false negative
        lo = box["lo"] + rng.normal(0.0, iou_threshold_noise, 3)
        hi = box["hi"] + rng.normal(0.0, iou_threshold_noise, 3)
        dets.append(
            {"box3d": np.concatenate([lo, hi]), "score": float(rng.uniform(0.55, 0.99)),
             "gt_id": box["oid"]}
        )
    if rng.random() < 0.35:
        lo = np.array([0.2, -0.3, 2.5]) + rng.normal(0.0, 0.1, 3)
        dets.append({"box3d": np.concatenate([lo, lo + 0.35]), "score": 0.42, "gt_id": -1})
    return dets


def _make_frame(cam: Camera, slice_name: str, frame_idx: int, seed: int) -> Frame:
    rng = np.random.default_rng(seed)
    left_boxes = _make_boxes(slice_name, frame_idx, right_view=False)
    right_boxes = _make_boxes(slice_name, frame_idx, right_view=True)
    rgb_l, depth, valid = _render(cam, left_boxes, 0.0, slice_name, rng)
    rgb_r, _, _ = _render(cam, right_boxes, cam.baseline, slice_name, rng)
    objects = [
        {"id": b["oid"], "box3d": np.concatenate([b["lo"], b["hi"]])}
        for b in left_boxes
        if b["oid"] >= 0
    ]
    return Frame(
        slice_name=slice_name,
        frame_idx=frame_idx,
        rgb_left=rgb_l,
        rgb_right=rgb_r,
        depth_gt=depth,
        valid_gt=valid,
        objects_gt=objects,
        detections=_noisy_detections(left_boxes, rng),
        metadata={"seed": seed},
    )


def build_dataset(cfg, seed: int | None = None) -> dict:
    """Deterministic synthetic stereo dataset with exact depth ground truth."""
    cam = Camera.from_config(cfg.camera)
    base = int(seed if seed is not None else cfg.seed)
    frames = []
    for s_idx, slice_name in enumerate(cfg.dataset.slices):
        for f in range(int(cfg.dataset.frames_per_slice)):
            frames.append(
                _make_frame(cam, slice_name, f, base + s_idx * 1000 + f * 17)
            )
    track_sequence = [
        _make_frame(cam, "dynamic", f, base + 90000 + f * 17)
        for f in range(int(cfg.dataset.validation_frames))
    ]
    return {"camera": cam, "frames": frames, "track_sequence": track_sequence}
