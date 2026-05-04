from __future__ import annotations

import csv
import html
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.config import DEMO_DATA_PATH, JIGSAW_TRAIN_PATH, LABELS

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
USER_RE = re.compile(r"(?<!\w)@[\w_]+")
SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

JIGSAW_COLUMNS = {
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
}


def clean_text(text: str) -> str:
    """Normalize text for modeling without removing moderation-relevant words."""
    if text is None:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = URL_RE.sub(" URL ", cleaned)
    cleaned = EMAIL_RE.sub(" EMAIL ", cleaned)
    cleaned = USER_RE.sub(" USER ", cleaned)
    cleaned = cleaned.replace("\u200b", " ")
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "non_toxic": "safe",
        "clean": "safe",
        "ok": "safe",
        "harassment": "abusive",
        "abuse": "abusive",
        "policy": "policy_violation",
        "policy_violating": "policy_violation",
        "violation": "policy_violation",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LABELS:
        raise ValueError(f"Unsupported label '{label}'. Expected one of: {', '.join(LABELS)}")
    return normalized


def read_labeled_csv(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        text_column = "text" if "text" in reader.fieldnames else "comment_text"
        if text_column not in reader.fieldnames:
            raise ValueError(f"{path} must include a 'text' or 'comment_text' column.")

        is_jigsaw = JIGSAW_COLUMNS.issubset(set(reader.fieldnames))
        for row in reader:
            text = clean_text(row.get(text_column, ""))
            if not text:
                continue
            if is_jigsaw:
                label = map_jigsaw_row(row)
                is_demo = "false"
            else:
                label = normalize_label(row.get("label", "safe"))
                is_demo = str(row.get("is_demo", "false")).lower()
            records.append({"text": text, "label": label, "is_demo": is_demo})
    if not records:
        raise ValueError(f"No usable records found in {path}.")
    return records


def map_jigsaw_row(row: dict[str, str]) -> str:
    scores = {column: int(float(row.get(column, 0) or 0)) for column in JIGSAW_COLUMNS}
    if sum(scores.values()) == 0:
        return "safe"
    if scores["threat"]:
        return "policy_violation"
    if scores["insult"] or scores["identity_hate"] or scores["severe_toxic"]:
        return "abusive"
    return "toxic"


def load_records(dataset_path: str | Path | None = None) -> list[dict[str, str]]:
    """Load local data, preferring explicit path, then Jigsaw, then demo data."""
    candidates = []
    if dataset_path:
        candidates.append(Path(dataset_path))
    candidates.extend([JIGSAW_TRAIN_PATH, DEMO_DATA_PATH])

    for candidate in candidates:
        if candidate.exists():
            return read_labeled_csv(candidate)
    raise FileNotFoundError(
        "No dataset found. Add Jigsaw train.csv under data/jigsaw/ or keep data/demo_toxicity.csv."
    )


def summarize_records(records: Iterable[dict[str, str]]) -> dict[str, object]:
    records = list(records)
    label_counts = Counter(record["label"] for record in records)
    return {
        "rows": len(records),
        "labels": {label: label_counts.get(label, 0) for label in LABELS},
        "demo_rows": sum(str(record.get("is_demo", "")).lower() == "true" for record in records),
    }


def stratified_split(
    records: list[dict[str, str]],
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label_records in grouped.values():
        label_records = list(label_records)
        rng.shuffle(label_records)
        n = len(label_records)
        train_end = max(1, int(round(n * train_size)))
        val_count = int(round(n * val_size))
        val_end = min(n, train_end + val_count)

        train.extend(label_records[:train_end])
        val.extend(label_records[train_end:val_end])
        test.extend(label_records[val_end:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_texts_labels(records: Iterable[dict[str, str]]) -> tuple[list[str], list[str]]:
    texts = [clean_text(record["text"]) for record in records]
    labels = [normalize_label(record["label"]) for record in records]
    return texts, labels
