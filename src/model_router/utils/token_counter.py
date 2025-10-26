"""Lightweight token counting heuristics."""

from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"\w+")


def count_tokens_approximate(text: str) -> int:
    """Approximate token count by combining words and symbol density."""

    if not text or not text.strip():
        return 0
    words = _WORD_PATTERN.findall(text)
    symbols = len(text) - len("".join(words))
    token_estimate = len(words) + int(symbols / 4)
    return max(1, token_estimate)
