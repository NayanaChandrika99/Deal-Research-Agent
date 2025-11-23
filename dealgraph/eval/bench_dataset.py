# ABOUTME: Bench dataset utilities for evaluating retrieval and reasoning quality.
# ABOUTME: Loads structured benchmark queries with labeled relevant deals.

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from ..config.settings import settings


class BenchQuery(BaseModel):
    """A labeled benchmark query with known relevant deals."""

    id: str
    text: str
    relevant_deal_ids: List[str]


def load_dealgraph_bench(path: Optional[str] = None) -> List[BenchQuery]:
    """
    Load the DealGraph Bench dataset.

    Args:
        path: Optional override path to the dataset JSON file.

    Returns:
        List of BenchQuery objects.
    """
    if path is None:
        path = settings.DATA_DIR / "bench" / "bench_queries.json"

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    return [BenchQuery(**entry) for entry in raw_entries]
