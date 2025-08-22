# visualization/utils.py
"""Utility functions for visualization."""

from typing import List


def _palette(n: int) -> List[str]:
    """Generate color palette for visualization."""
    base = [
        "#64FFDA",
        "#7C4DFF",
        "#FF6B6B",
        "#FFCA28",
        "#29B6F6",
        "#66BB6A",
        "#F06292",
        "#26A69A",
        "#AB47BC",
        "#FFA726",
    ]
    return [base[i % len(base)] for i in range(n)]
