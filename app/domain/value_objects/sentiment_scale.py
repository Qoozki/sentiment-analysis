"""The sentiment scales and the rule that turns a score into a Positive / Neutral / Negative label."""
from __future__ import annotations

LEVELS5 = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
WEIGHTS5 = [-1.0, -0.5, 0.0, 0.5, 1.0]
LEVELS3 = ["Negative", "Neutral", "Positive"]
WEIGHTS3 = [-1.0, 0.0, 1.0]

LABELS = ["Positive", "Neutral", "Negative"]
LEVEL_ORDER = LEVELS5
THRESHOLD = 0.15  # |score| below this is reported as Neutral in sentiment_label


def label_for(score: float) -> str:
    if score >= THRESHOLD:
        return "Positive"
    if score <= -THRESHOLD:
        return "Negative"
    return "Neutral"
