"""
Normalize bullet-style resume text into continuous paragraphs for agents / LLMs.

Kept separate from ``agent1_scrubber`` so formatting is optional and testable.
"""

from __future__ import annotations

import re
from typing import List

# Line starts like: • item, - item, * item, 1. item, 1) item (after whitespace)
_BULLET_LINE = re.compile(
    r"^\s*(?:[•·▪‣◦]\s*|[\-*+]\s+|\d{1,3}[).]\s+)"
)


def _ensure_sentence_end(fragment: str) -> str:
    s = fragment.strip()
    if not s:
        return ""
    if s[-1] in ".!?…:;":
        return s
    return s + "."


def _join_bullet_fragments(fragments: List[str]) -> str:
    parts: List[str] = []
    for raw in fragments:
        s = _ensure_sentence_end(raw)
        if s:
            parts.append(s)
    return " ".join(parts)


def bullets_to_prose(text: str) -> str:
    """
    Turn bullet lists into flowing paragraphs; keep section blocks separated by
    blank lines.

    Lines that match a bullet prefix are grouped; each group becomes one
    paragraph (fragments separated by spaces, with light sentence punctuation).
    Non-bullet blocks (headers, paragraphs) are merged line-by-line with spaces
    inside the block; blank lines separate blocks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return text.strip()

    lines = text.split("\n")

    if not any(_BULLET_LINE.match(L) for L in lines if L.strip()):
        return text.strip()

    paragraphs: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if _BULLET_LINE.match(line):
            group: List[str] = []
            while i < n and lines[i].strip():
                if not _BULLET_LINE.match(lines[i]):
                    break
                body = _BULLET_LINE.sub("", lines[i], count=1).strip()
                if body:
                    group.append(body)
                i += 1
            merged = _join_bullet_fragments(group)
            if merged:
                paragraphs.append(merged)
            continue

        block: List[str] = []
        while i < n and lines[i].strip():
            if _BULLET_LINE.match(lines[i]):
                break
            block.append(lines[i].strip())
            i += 1
        paragraphs.append(" ".join(block))

    return "\n\n".join(p for p in paragraphs if p.strip())
