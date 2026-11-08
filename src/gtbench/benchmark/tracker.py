from __future__ import annotations

import numpy as np

from .metrics import box3d_iou


class IoU3DTracker:
    """Minimal tracking-by-detection tracker with greedy 3D-IoU association.

    Deliberately transparent: it exists so the tracking benchmark measures a
    real association module against real 3D box ground truth, not a lookup.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        pairs = []
        for tid, tr in self.tracks.items():
            if tr["age"] > 0:
                continue
            for di, det in enumerate(detections):
                iou = box3d_iou(tr["box3d"], det["box3d"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for iou, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            self.tracks[tid]["box3d"] = np.asarray(detections[di]["box3d"])
            self.tracks[tid]["age"] = 0

        for tid, tr in self.tracks.items():
            if tid not in used_t:
                tr["age"] += 1
        self.tracks = {t: tr for t, tr in self.tracks.items() if tr["age"] <= self.max_age}

        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks[self.next_id] = {"box3d": np.asarray(det["box3d"]), "age": 0}
                self.next_id += 1

        return [
            {"id": tid, "box3d": tr["box3d"]}
            for tid, tr in self.tracks.items()
            if tr["age"] == 0
        ]
