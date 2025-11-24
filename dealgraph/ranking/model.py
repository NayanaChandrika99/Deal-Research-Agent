# ABOUTME: Gradient boosting-based ranking model.
# ABOUTME: Wraps scikit-learn estimator with save/load helpers and ranking API.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ..data.schemas import CandidateDeal, RankedDeal
from .features import build_feature_matrix, candidate_to_features


class DealRanker:
    """Lightweight wrapper around a regression model to score deals."""

    def __init__(self, model: Optional[GradientBoostingRegressor] = None):
        self.model = model or GradientBoostingRegressor(random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DealRanker":
        self.model.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def rank(self, candidates: Iterable[CandidateDeal]) -> List[RankedDeal]:
        candidates = list(candidates)
        if not candidates:
            return []
        X = build_feature_matrix(candidates)
        scores = self.predict_scores(X)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        ranked_deals: List[RankedDeal] = []
        for idx, (candidate, score) in enumerate(ranked, start=1):
            ranked_deals.append(
                RankedDeal(candidate=candidate, score=float(score), rank=idx)
            )
        return ranked_deals

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "DealRanker":
        model = joblib.load(path)
        return cls(model=model)
