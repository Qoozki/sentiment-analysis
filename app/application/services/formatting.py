"""Small text-formatting helpers shared by the insight builders."""
from __future__ import annotations


def signed(x: float, d: int = 2) -> str:
    """+0.35 / \u22120.35 (a real minus sign)."""
    return f"{x:+.{d}f}".replace("-", "\u2212")
