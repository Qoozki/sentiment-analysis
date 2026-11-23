"""What an engine returns for one text."""
from __future__ import annotations

from typing import NamedTuple


class Prediction(NamedTuple):
    detail: str                      # Very Negative ... Very Positive (or Negative / Neutral / Positive)
    score: float                     # -1 .. +1
    confidence: float                # probability of the predicted level
    extras: dict | None = None       # extra columns an engine may add, e.g. english_text
