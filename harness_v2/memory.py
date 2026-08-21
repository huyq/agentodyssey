"""Append-only episodic memory with keyword search (harness_v2).

Deliberately minimal: no embedder and no index structures beyond the raw
entry list.  Entries carry the game step at which they were written so that
search results can be located in time.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from utils import atomic_write

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class KeywordMemory:
    """Append-only text memory retrievable by keyword overlap."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, step: int, text: str) -> None:
        self.entries.append({"step": int(step), "text": text})

    def search(self, query: str, top_k: int = 3, exclude_last: int = 0) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` entries overlapping with the query terms.

        Scoring: number of distinct query terms hit (primary), total term
        frequency of the hit terms (secondary), recency by step (tiebreak).
        Entries with zero overlap are never returned.  ``exclude_last`` skips
        the most recent N entries (e.g. those already shown as STM context).
        """
        terms = set(_tokens(query))
        pool = self.entries[:-int(exclude_last)] if exclude_last > 0 else self.entries
        if not terms or not pool:
            return []
        scored = []
        for entry in pool:
            entry_tokens = _tokens(entry["text"])
            hits = terms & set(entry_tokens)
            if not hits:
                continue
            tf = sum(entry_tokens.count(t) for t in hits)
            scored.append((len(hits), tf, int(entry["step"]), entry))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [entry for *_, entry in scored[: max(1, int(top_k))]]

    def to_dict(self) -> dict:
        return {"entries": list(self.entries)}

    def load_from_dict(self, d: dict) -> None:
        self.entries = [
            {"step": int(e.get("step", 0)), "text": str(e.get("text", ""))}
            for e in (d.get("entries", []) or [])
        ]

    def save(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
