import numpy as np

from gtbench.data.calibration import Camera
from gtbench.geometry import (
    box_sum,
    epipolar_residual,
    essential_matrix,
    fit_scale_shift,
    triangulate_dlt,
)


def _cam():
    return Camera(320, 192, 250.0, 250.0, 160.0, 96.0, 0.24)


def test_triangulation_roundtrip():
    cam = _cam()
    P1, P2 = cam.projection_left(), cam.projection_right()
    X = np.array([[0.3, -0.2, 3.0], [1.0, 0.4, 6.5], [-0.7, 0.1, 2.0]])
    x1 = cam.project(X)
    Xr = X.copy()
    Xr[:, 0] -= cam.baseline
    x2 = cam.project(Xr)
    Xhat = triangulate_dlt(P1, P2, x1, x2)
    assert np.allclose(Xhat, X, atol=1e-5)


def test_epipolar_constraint_holds():
    cam = _cam()
    X = np.array([[0.3, -0.2, 3.0], [1.0, 0.4, 6.5], [-0.7, 0.1, 2.0]])
    x1 = cam.project(X)
    Xr = X.copy()
    Xr[:, 0] -= cam.baseline
    x2 = cam.project(Xr)
    E = essential_matrix(np.eye(3), np.array([-cam.baseline, 0.0, 0.0]))
    n1 = (x1 - np.array([cam.cx, cam.cy])) / np.array([cam.fx, cam.fy])
    n2 = (x2 - np.array([cam.cx, cam.cy])) / np.array([cam.fx, cam.fy])
    assert np.all(epipolar_residual(n1, n2, E) < 1e-6)


def test_box_sum_on_constant_image():
    a = np.ones((10, 10))
    assert np.allclose(box_sum(a, 1), 9.0)
    assert np.allclose(box_sum(a, 2), 25.0)


def test_disparity_depth_inverse():
    cam = _cam()
    z = np.array([2.0, 8.0])
    d = cam.disparity_from_depth(z)
    assert np.allclose(cam.depth_from_disparity(d), z)
    assert d[0] > d[1]


def test_scale_shift_fit():
    x = np.linspace(1.0, 5.0, 50)
    y = 3.0 * x + 2.0
    a, b = fit_scale_shift(x, y, np.ones(50, dtype=bool))
    assert abs(a - 3.0) < 1e-6 and abs(b - 2.0) < 1e-6
