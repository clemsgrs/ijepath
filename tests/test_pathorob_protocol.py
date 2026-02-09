import numpy as np

from ijepath.eval.pathorob.apd import _apd_from_split_acc
from ijepath.eval.pathorob.clustering import clustering_score
from ijepath.eval.pathorob.ri import _normalized_ri_from_neighbors


def test_ri_formula_returns_expected_extreme():
    labels = np.array([0, 0, 1, 1])
    centers = np.array([0, 1, 0, 1])
    neigh = np.array([[1], [0], [3], [2]], dtype=int)
    assert _normalized_ri_from_neighbors(labels, centers, neigh, k=1) == 1.0


def test_clustering_score_formula_prefers_biology_over_center():
    pred = np.array([0, 0, 1, 1])
    bio = np.array([0, 0, 1, 1])
    center = np.array([0, 1, 0, 1])
    assert clustering_score(pred, bio, center) > 0.9


def test_apd_formula_matches_relative_drop_mean():
    split_acc = {
        1: 0.80,
        2: 0.76,
        3: 0.72,
    }
    apd = _apd_from_split_acc(split_acc)
    assert abs(apd - (-0.075)) < 1e-8
