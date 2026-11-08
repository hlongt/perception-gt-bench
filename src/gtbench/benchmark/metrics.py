from __future__ import annotations

from collections import defaultdict

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """Standard depth-evaluation metrics on valid pixels only."""
    p = np.asarray(pred, dtype=np.float64)[valid]
    g = np.asarray(gt, dtype=np.float64)[valid]
    if len(g) == 0:
        return {"absrel": None, "rmse": None, "silog": None, "delta_1_25": None,
                "delta_1_25_2": None, "delta_1_25_3": None, "a1": None,
                "p50_absrel": None, "p90_absrel": None, "p99_absrel": None, "n": 0}
    g = np.maximum(g, 1e-6)
    p = np.maximum(p, 1e-6)
    absrel = np.abs(p - g) / g
    logdiff = np.log(p) - np.log(g)
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": round(float(absrel.mean()), 5),
        "rmse": round(float(np.sqrt(((p - g) ** 2).mean())), 5),
        "silog": round(float(np.sqrt((logdiff**2).mean() - logdiff.mean() ** 2) * 100), 5),
        "delta_1_25": round(float((ratio < 1.25).mean()), 5),
        "delta_1_25_2": round(float((ratio < 1.25**2).mean()), 5),
        "delta_1_25_3": round(float((ratio < 1.25**3).mean()), 5),
        "a1": round(float((ratio < 1.25).mean()), 5),
        "p50_absrel": round(float(np.percentile(absrel, 50)), 5),
        "p90_absrel": round(float(np.percentile(absrel, 90)), 5),
        "p99_absrel": round(float(np.percentile(absrel, 99)), 5),
        "n": int(len(g)),
    }


def box3d_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two axis-aligned 3D boxes encoded as [x1,y1,z1,x2,y2,z2]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    inter = float(np.prod(np.maximum(hi - lo, 0.0)))
    va = float(np.prod(np.maximum(a[3:] - a[:3], 0.0)))
    vb = float(np.prod(np.maximum(b[3:] - b[:3], 0.0)))
    union = va + vb - inter
    return inter / union if union > 1e-9 else 0.0


def _greedy_match(gt_boxes, pred_boxes, thr):
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            iou = box3d_iou(gb, pb)
            if iou >= thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    ug, up, matched = set(), set(), []
    for iou, gi, pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi)
            up.add(pi)
            matched.append((gi, pi, iou))
    return matched


def tracking_metrics(sequence, iou_threshold: float = 0.25) -> dict:
    """MOTA / MOTP / IDF1 / IDSW + single-threshold HOTA components.

    sequence: list of (gt_list, pred_list); each item is {"id", "box3d"}.
    """
    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    prev = {}
    pair_tp = defaultdict(int)
    gt_tp = defaultdict(int)
    pred_tp = defaultdict(int)

    for gt_list, pred_list in sequence:
        gt_boxes = [g["box3d"] for g in gt_list]
        pred_boxes = [p["box3d"] for p in pred_list]
        m = _greedy_match(gt_boxes, pred_boxes, iou_threshold)
        cur = {}
        for gi, pi, iou in m:
            gid, pid = gt_list[gi]["id"], pred_list[pi]["id"]
            cur[pid] = gid
            if pid in prev and prev[pid] != gid:
                idsw += 1
            iou_sum += iou
            pair_tp[(gid, pid)] += 1
            gt_tp[gid] += 1
            pred_tp[pid] += 1
        tp += len(m)
        fp += len(pred_boxes) - len(m)
        fn += len(gt_boxes) - len(m)
        prev = cur

    mota = 1.0 - (fn + fp + idsw) / max(tp + fn, 1)
    motp = iou_sum / tp if tp else 0.0
    id_pairs = [(g, p) for (g, p) in pair_tp]
    idtp = 0.0
    for gid, pid in id_pairs:
        idtp += pair_tp[(gid, pid)]
    idf1 = 2 * idtp / max(2 * idtp + fp + fn, 1)

    det_a = tp / max(tp + fp + fn, 1)
    ass_scores = []
    for gid in gt_tp:
        s = 0.0
        for (g, p), c in pair_tp.items():
            if g != gid:
                continue
            s += c / max(c + (pred_tp[p] - c) + (gt_tp[g] - c), 1)
        ass_scores.append(s)
    ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
    hota = float(np.sqrt(max(det_a * ass_a, 0.0)))

    return {
        "MOTA": round(float(mota), 4),
        "MOTP": round(float(motp), 4),
        "IDF1": round(float(idf1), 4),
        "IDSW": int(idsw),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "DetA": round(float(det_a), 4),
        "AssA": round(float(ass_a), 4),
        "HOTA_single_threshold": round(hota, 4),
        "frames": len(sequence),
    }
