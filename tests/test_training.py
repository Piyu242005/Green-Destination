import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import select_threshold


def test_threshold_selection_is_bounded():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.35, 0.55, 0.70, 0.90])
    threshold, score = select_threshold(y_true, probabilities)
    assert 0.10 <= threshold <= 0.90
    assert 0.0 <= score <= 1.0
