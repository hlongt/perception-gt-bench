import numpy as np

from gtbench.benchmark.metrics import box3d_iou, depth_metrics, tracking_metrics


def test_perfect_depth_prediction():
    gt = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    m = depth_metrics(gt.copy(), gt, np.ones_like(gt, dtype=bool))
    assert m["absrel"] == 0.0
    assert m["rmse"] == 0.0
    assert m["delta_1_25"] == 1.0
    assert m["n"] == 4


def test_absrel_known_value():
    gt = np.array([[2.0, 4.0]], dtype=np.float32)
    pred = gt * 2.0
    m = depth_metrics(pred, gt, np.ones_like(gt, dtype=bool))
    assert abs(m["absrel"] - 1.0) < 1e-6
    assert m["delta_1_25"] == 0.0


def test_box3d_iou():
    a = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    assert abs(box3d_iou(a, a) - 1.0) < 1e-9
    b = np.array([5, 5, 5, 6, 6, 6], dtype=float)
    assert box3d_iou(a, b) == 0.0
    c = np.array([0.5, 0, 0, 1.5, 1, 1], dtype=float)
    assert abs(box3d_iou(a, c) - 0.5 / 1.5) < 1e-9


def test_tracking_perfect_sequence():
    box = np.array([0, 0, 2, 1, 1, 3], dtype=float)
    seq = [([{"id": 7, "box3d": box}], [{"id": 7, "box3d": box}]) for _ in range(4)]
    m = tracking_metrics(seq, 0.25)
    assert m["MOTA"] == 1.0
    assert m["IDF1"] == 1.0
    assert m["TP"] == 4 and m["FP"] == 0 and m["FN"] == 0
    assert m["IDSW"] == 0


def test_tracking_detects_false_positives():
    box = np.array([0, 0, 2, 1, 1, 3], dtype=float)
    far = np.array([9, 9, 9, 10, 10, 10], dtype=float)
    seq = [([{"id": 1, "box3d": box}],
            [{"id": 1, "box3d": box}, {"id": 2, "box3d": far}])]
    m = tracking_metrics(seq, 0.25)
    assert m["FP"] == 1
    assert m["MOTA"] < 1.0
