"""Resignation-threshold calibration from playthrough audits (pure, testable)."""

from __future__ import annotations


def calibrate_resign(threshold: float, would: int, false: int, target: float) -> float:
    """New resign threshold from a window of playthrough audits.

    `would` = would-be resignations seen in playthrough games, `false` = how many
    of those sides did not go on to lose. Too many wrong resignations -> resign
    only at a lower expected score; comfortably under target -> allow resigning
    slightly earlier (shorter games, less compute on dead positions)."""
    fp = false / max(1, would)
    if fp > target:
        return max(0.005, threshold * 0.8)
    if fp < 0.5 * target:
        return min(0.2, threshold * 1.1)
    return threshold
