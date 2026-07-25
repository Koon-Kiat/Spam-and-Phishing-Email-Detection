import numpy as np

from spamandphishingdetection.evaluation import (
    evaluate_probabilities,
    tune_threshold,
)


def test_threshold_tuning_and_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, selection = tune_threshold(labels, probabilities)
    metrics = evaluate_probabilities(labels, probabilities, threshold)

    assert 0.4 < threshold <= 0.6
    assert selection["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["phishing_recall"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
