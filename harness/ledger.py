"""Structured, episode-local state ledger.

The parser intentionally extracts only facts already stated by observations;
it does not inspect game definitions or promote model reasoning to a fact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


_NO_VALUE = {"", "none", "nothing", "no objects", "no one", "nobody", "unknown"}
_NUMBER = r"[-+]?\d+(?:\.\d+)?"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _split_values(value: str) -> list[str]:
    value = _clean(value).rstrip(".")
    if value.casefold() in _NO_VALUE:
        return []
    return [part.strip() for part in re.split(r"\s*,\s*|\s+and\s+", value) if part.strip()]


def _parse_inventory(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in _split_values(value):
        match = re.match(r"^(\d+)\s+(.+)$", item)
        if match:
            count, name = int(match.group(1)), _clean(match.group(2))
        else:
            count, name = 1, item
        result[name] = result.get(name, 0) + count
    return result


def _parse_names(value: str) -> list[str]:
    """Parse visible entities while dropping observation-only quantity prefixes."""
    return list(_parse_inventory(value).keys())


def _sentence_value(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text, re.I | re.M)
    return _clean(match.group(1)) if match else None


@dataclass
class Fact:
    claim: str
    confidence: float = 1.0
    evidence_step: Optional[int] = None
    last_verified_step: Optional[int] = None
    source: str = "observation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        return cls(
            claim=str(data.get("claim", "")),
            confidence=float(data.get("confidence", 1.0)),
            evidence_step=data.get("evidence_step"),
            last_verified_step=data.get("last_verified_step"),
            source=str(data.get("source", "observation")),
        )


@dataclass
class StateLedger:
    """Facts and progress known during one game episode."""

    goal: dict[str, Any] = field(default_factory=lambda: {"main_stage": "unknown", "subgoal": ""})
    location: dict[str, Any] = field(default_factory=lambda: {"area": "", "place": "", "neighbors": []})
    inventory: dict[str, Any] = field(default_factory=lambda: {"items": {}, "equipped": []})
    entities: dict[str, Any] = field(default_factory=lambda: {"visible_objects": [], "visible_npcs": []})
    vitals: dict[str, Any] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    step: int = -1

    def reset(self) -> None:
        self.goal = {"main_stage": "unknown", "subgoal": ""}
        self.location = {"area": "", "place": "", "neighbors": []}
        self.inventory = {"items": {}, "equipped": []}
        self.entities = {"visible_objects": [], "visible_npcs": []}
        self.vitals = {}
        self.facts.clear()
        self.recent_failures.clear()
        self.skills.clear()
        self.progress.clear()
        self.step = -1

    def update_observation(self, observation: Any, step: Optional[int] = None) -> "StateLedger":
        text = str(observation or "")
        if step is not None:
            self.step = int(step)
        current_time = _sentence_value(text, r"^\s*Current Time\s*:\s*(.+?)\s*$")
        current_location = _sentence_value(text, r"^\s*Current Location\s*:\s*(.+?)\s*$")
        holding = _sentence_value(text, r"^\s*(?:I|We|You)\s+(?:am|are)\s+holding\s+(.+?)\.?\s*$")
        equipped = _sentence_value(text, r"^\s*(?:I|We|You)\s+have\s+equipped\s+(.+?)\.?\s*$")
        neighboring = _sentence_value(text, r"^\s*Neighboring areas\s*:\s*(.+?)\s*$")

        if current_time is not None:
            self.vitals["time"] = current_time
        if current_location is not None:
            parts = [part.strip() for part in current_location.split(",", 1)]
            self.location["place"] = parts[0]
            self.location["area"] = parts[1] if len(parts) > 1 else parts[0]
        if holding is not None:
            self.inventory["items"] = _parse_inventory(holding)
        if equipped is not None:
            self.inventory["equipped"] = _parse_names(equipped)
        if neighboring is not None:
            self.location["neighbors"] = _split_values(neighboring)

        # The observation uses first-person possessives in the default env;
        # accept equivalent generated phrasing as well.
        stat_patterns = {
            "level": r"(?:My|Your|Their)\s+level\s+is\s+(%s)" % _NUMBER,
            "attack": r"(?:My|Your|Their)\s+attack\s+is\s+at\s+(%s)" % _NUMBER,
            "defense": r"(?:My|Your|Their)\s+defense\s+is\s+at\s+(%s)" % _NUMBER,
            "health": r"(?:My|Your|Their)\s+health\s+is\s+at\s+(%s)" % _NUMBER,
            "experience": r"(?:My|Your|Their)\s+experience\s+is\s+at\s+(%s)" % _NUMBER,
        }
        for key, pattern in stat_patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                raw = match.group(1)
                self.vitals[key] = float(raw) if "." in raw else int(raw)

        # Object/NPC lines are deliberately conservative: only the canonical
        # observation clauses are recorded, never arbitrary reasoning text.
        object_match = re.search(r"(?im)^\s*(?:I|We|You)\s+see\s+(.+?)\s+near\s+(?:me|us|you)\.?", text)
        npc_match = re.search(r"(?im)^\s*(?:I|We|You)\s+see\s+(.+?)\s+nearby\.?", text)
        if object_match:
            self.entities["visible_objects"] = _parse_names(object_match.group(1))
        if npc_match:
            self.entities["visible_npcs"] = _parse_names(npc_match.group(1))

        quest_match = re.search(r"(?im)^\s*===\s*MAIN QUEST:\s*(.+?)\s*===\s*$", text)
        chapter_match = re.search(r"(?im)^\s*Chapter\s+([^:]+):\s*(.+?)\s*$", text)
        if quest_match:
            self.goal["main_quest"] = _clean(quest_match.group(1))
        if chapter_match:
            self.goal["main_stage"] = _clean(chapter_match.group(1))
            self.goal["stage_title"] = _clean(chapter_match.group(2))

        # Tutorial directive: "Required action: `enter hall`".  Only the
        # tutorial emits this exact format; clear it as soon as observations
        # stop carrying it so it is never stale outside the tutorial.
        required_action = _sentence_value(text, r"^\s*Required action\s*:\s*`([^`]+)`\s*$")
        if required_action is not None:
            self.goal["required_action"] = required_action
        elif "required_action" in self.goal:
            del self.goal["required_action"]

        return self

    def record_fact(
        self,
        claim: str,
        *,
        confidence: float = 1.0,
        step: Optional[int] = None,
        source: str = "observation",
    ) -> Fact:
        normalized = _clean(claim)
        if not normalized:
            raise ValueError("claim must not be empty")
        confidence = max(0.0, min(1.0, float(confidence)))
        for fact in self.facts:
            if fact.claim.casefold() == normalized.casefold():
                fact.confidence = max(fact.confidence, confidence)
                fact.last_verified_step = self.step if step is None else int(step)
                return fact
        fact = Fact(normalized, confidence, step, step, source)
        self.facts.append(fact)
        return fact

    def record_failure(self, action: Any, feedback: Any = "", step: Optional[int] = None) -> None:
        item = {"action": _clean(action), "feedback": _clean(feedback), "step": self.step if step is None else int(step)}
        if item not in self.recent_failures:
            self.recent_failures.append(item)
        self.recent_failures[:] = self.recent_failures[-16:]

    def update_progress(self, reward: Mapping[str, Any] | Any) -> None:
        values = vars(reward) if hasattr(reward, "__dict__") else (reward if isinstance(reward, Mapping) else {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.progress[key] = self.progress.get(key, 0) + value

    def summary(self) -> str:
        """Compact deterministic summary suitable for an event or prompt."""
        return (
            f"location={self.location.get('place', '')}, {self.location.get('area', '')}; "
            f"neighbors={', '.join(self.location.get('neighbors', [])) or 'none'}; "
            f"items={self.inventory.get('items', {}) or 'none'}; "
            f"equipped={', '.join(self.inventory.get('equipped', [])) or 'none'}; "
            f"main_stage={self.goal.get('main_stage', 'unknown')}; step={self.step}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["facts"] = [fact.to_dict() for fact in self.facts]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateLedger":
        ledger = cls()
        for key in ("goal", "location", "inventory", "entities", "vitals", "recent_failures", "skills", "progress"):
            if key in data:
                setattr(ledger, key, data[key])
        ledger.facts = [Fact.from_dict(item) for item in data.get("facts", []) or []]
        ledger.step = int(data.get("step", -1))
        return ledger
