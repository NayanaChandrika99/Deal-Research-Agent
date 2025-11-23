"""Unit tests for evaluation metric helpers."""

import pytest

from dealgraph.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k_basic():
    predicted = ["d1", "d2", "d3"]
    relevant = ["d3", "d4"]
    assert precision_at_k(predicted, relevant, k=2) == 0.0
    assert precision_at_k(predicted, relevant, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_basic():
    predicted = ["a", "b", "c", "d"]
    relevant = ["b", "d"]
    assert recall_at_k(predicted, relevant, k=1) == 0.0
    assert recall_at_k(predicted, relevant, k=4) == 1.0


def test_ndcg_at_k_perfect_and_partial():
    relevant = ["x1", "x2", "x3"]
    perfect = ["x1", "x2", "x3"]
    partial = ["x3", "x4", "x1"]
    
    assert ndcg_at_k(perfect, relevant, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(partial, relevant, k=3) < 1.0
