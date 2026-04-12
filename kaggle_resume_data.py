"""
Load resume plain-text samples from Kaggle via ``kagglehub``.

Default dataset: ``snehaanbhawal/resume-dataset`` (CSVs often use ``Resume_str``
for the resume body plus a category column).

Auth: create ``~/.kaggle/kaggle.json`` from Kaggle Account → API, or set
``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` in the environment.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, NamedTuple


class ResumeRow(NamedTuple):
    """One resume-like row parsed from a Kaggle CSV."""

    source_name: str
    row_index: int
    category: str | None
    text: str


def download_resume_dataset(slug: str = "snehaanbhawal/resume-dataset") -> Path:
    """Download (or return cache path for) a Kaggle dataset directory."""
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install kagglehub: pip install kagglehub"
        ) from e

    path = kagglehub.dataset_download(slug)
    return Path(path)


def _guess_text_column(fieldnames: list[str]) -> str | None:
    if not fieldnames:
        return None
    # Normalize headers so ``Resume_str`` / odd spacing still map to lookup keys.
    lower = {f.lstrip("\ufeff").lower().strip(): f for f in fieldnames}
    for key in (
        "resume_str",  # snehaanbhawal/resume-dataset
        "resume",
        "resume data",
        "resume text",
        "text",
        "content",
        "cv",
        "raw resume",
    ):
        if key in lower:
            return lower[key]
    for f in fieldnames:
        compact = f.lower().replace(" ", "")
        if "resume" in compact and "category" not in compact:
            return f
    return None


def _guess_category_column(fieldnames: list[str]) -> str | None:
    lower = {f.lstrip("\ufeff").lower().strip(): f for f in fieldnames}
    for key in ("category", "label", "class", "job category"):
        if key in lower:
            return lower[key]
    return None


def iter_resume_rows_from_csv(
    csv_path: Path,
    *,
    text_col: str | None = None,
    category_col: str | None = None,
    max_rows: int | None = None,
    min_chars: int = 40,
) -> Iterator[ResumeRow]:
    """Yield rows from a CSV that look like full resume text."""
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        names = list(reader.fieldnames)
        tc = text_col or _guess_text_column(names)
        if not tc:
            return
        cc = category_col or _guess_category_column(names)
        yielded = 0
        for i, row in enumerate(reader):
            if max_rows is not None and yielded >= max_rows:
                break
            raw = (row.get(tc) or "").strip()
            if len(raw) < min_chars:
                continue
            cat = (row.get(cc) or "").strip() if cc else None
            yield ResumeRow(csv_path.name, i, cat or None, raw)
            yielded += 1


def _csv_priority(path: Path) -> tuple[int, int]:
    """Higher sort key = try this file first."""
    name = path.name.lower()
    tier = 0
    if "updated" in name and "resume" in name:
        tier = 4
    elif "resume" in name and "skill" not in name:
        tier = 3
    elif "resume" in name:
        tier = 2
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (tier, size)


def sorted_resume_csvs(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*.csv") if p.is_file()]
    paths.sort(key=_csv_priority, reverse=True)
    return paths


def load_labeled_resume_texts(
    root: Path | str,
    *,
    max_rows: int = 300,
    max_rows_per_csv: int = 150,
) -> list[tuple[str, str]]:
    """
    Build (label, text) pairs for UI pickers.

    *label* is a short human-readable summary; *text* is the resume body.
    """
    root = Path(root)
    out: list[tuple[str, str]] = []
    for csv_path in sorted_resume_csvs(root):
        n = 0
        for row in iter_resume_rows_from_csv(
            csv_path, max_rows=max_rows_per_csv * 3
        ):
            cat = row.category or "Uncategorized"
            label = f"{cat} · {row.source_name} · row {row.row_index}"
            out.append((label, row.text))
            n += 1
            if len(out) >= max_rows or n >= max_rows_per_csv:
                break
        if len(out) >= max_rows:
            break
    return out
