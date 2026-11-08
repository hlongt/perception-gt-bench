from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def frame_digest(frame) -> str:
    """Content hash of a frame, used for de-duplication and resume.

    Mirrors the file_hash + checkpoint pattern used in production data loops:
    re-running the pipeline never processes the same frame twice.
    """
    h = hashlib.sha256()
    h.update(frame.slice_name.encode("utf-8"))
    h.update(str(frame.frame_idx).encode("utf-8"))
    for arr in (frame.rgb_left, frame.rgb_right, frame.depth_gt):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def frame_key(frame) -> str:
    return f"{frame.slice_name}/{frame.frame_idx:04d}"


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def seen(self, frame) -> bool:
        rec = self.records.get(frame_key(frame))
        return bool(rec and rec.get("digest") == frame_digest(frame))

    def mark(self, frame, **extra):
        self.records[frame_key(frame)] = {
            "digest": frame_digest(frame),
            "slice": frame.slice_name,
            "frame_idx": frame.frame_idx,
            **extra,
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class KITTIAdapter:
    """Adapter boundary for real KITTI / KITTI-360 data.

    The synthetic renderer makes the whole repository reproducible without a
    registered KITTI account. To run the identical pipeline on real data,
    implement ``load_frame`` so it returns the same ``Frame`` dataclass:

      - read ``image_2`` / ``image_3`` (rectified left/right PNG),
      - read ``calib_cam_to_cam.txt`` -> f, cx, cy, baseline,
      - project ``velodyne`` points into the left image to build a sparse
        depth/GT map (or use the KITTI depth completion GT),
      - read ``label_2`` tracking annotations for 3D boxes.

    Every downstream module (pseudo_gt, gt_qc, benchmark) consumes only the
    Frame interface, so no other code changes are required.
    """

    @staticmethod
    def load_frame(root: str, drive: str, index: int):
        root_p = Path(root)
        if not root_p.exists():
            raise FileNotFoundError(
                f"KITTI root not found: {root_p}. Download KITTI raw / object "
                "data first, or use the bundled synthetic dataset."
            )
        raise NotImplementedError(
            "KITTIAdapter.load_frame is an integration point; see class docstring "
            "for the exact field mapping."
        )
