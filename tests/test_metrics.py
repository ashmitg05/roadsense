import numpy as np

from src.eval.metrics import box_iou, evaluate, match


def test_box_iou() -> None:
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_perfect_predictions_give_map_1() -> None:
    gt = np.array([[0, 0, 0, 10, 10], [3, 20, 20, 40, 40]], float)
    pred = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 3]], float)
    r = evaluate([pred], [gt], nc=4)
    assert r["map50"] == 1.0 and r["map50_95"] == 1.0


def test_wrong_class_and_duplicate_are_fp() -> None:
    gt = np.array([[0, 0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10, 0.9, 1], [0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.7, 0]], float)
    tp = match(pred, gt)
    assert tp[:, 0].tolist() == [False, True, False]


def test_missed_half_gives_ap_about_half() -> None:
    gt = np.array([[0, 0, 0, 10, 10], [0, 50, 50, 60, 60]], float)
    pred = np.array([[0, 0, 10, 10, 0.9, 0]], float)
    r = evaluate([pred], [gt], nc=1)
    assert abs(r["map50"] - 51 / 101) < 1e-6  # recall 0.5 -> 51 of 101 grid points hit
