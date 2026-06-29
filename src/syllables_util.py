"""English syllable counting for 5-7-5 haiku form scoring."""

from __future__ import annotations

import re

import syllables

from src.schema import TARGET_SYLLABLES

_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)


def count_syllables(text: str) -> int:
    """Estimate syllables in a line (punctuation stripped, words summed)."""
    cleaned = _PUNCT_RE.sub(" ", text).strip()
    if not cleaned:
        return 0
    total = 0
    for word in cleaned.split():
        w = word.strip("-'")
        if not w:
            continue
        total += max(1, syllables.estimate(w))
    return total


def line_syllable_counts(lines: list[str]) -> list[int]:
    return [count_syllables(line) for line in lines]


def syllable_l1_error(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> int:
    counts = line_syllable_counts(lines)
    if len(counts) != len(target):
        return sum(abs(c - t) for c, t in zip(counts, target, strict=False)) + 50
    return sum(abs(c - t) for c, t in zip(counts, target, strict=True))


def syllable_perfect(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> bool:
    counts = line_syllable_counts(lines)
    return len(counts) == len(target) and counts == list(target)
