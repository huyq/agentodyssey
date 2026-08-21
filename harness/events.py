"""Current-episode interaction event log used by the harness."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Iterable, Mapping, Optional

_STEP_SUFFIX = re.compile(r";\s*step=-?\d+\s*$")


def _strip_step(summary: str) -> str:
    """State summaries embed the step counter; strip it for change detection."""
    return _STEP_SUFFIX.sub("", summary)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            str(key): val
            for key, val in vars(value).items()
            if not str(key).startswith("_")
        }
    return {}


def reward_delta(value: Any) -> dict[str, float]:
    """Convert a RewardBreakdown or mapping to JSON-safe numeric fields."""

    result: dict[str, float] = {}
    for key, item in _as_dict(value).items():
        if isinstance(item, bool):
            result[key] = float(item)
        elif isinstance(item, (int, float)):
            result[key] = float(item) if isinstance(item, float) else int(item)
    return result


def classify_outcome(
    reward: Any = None,
    feedback: Any = "",
    invalid: bool = False,
    events: Optional[Iterable[Any]] = None,
) -> str:
    """Classify an interaction using environment signals, conservatively."""

    if invalid:
        return "invalid"
    event_types: set[str] = set()
    for event in events or ():
        event_type = event.get("type", "") if isinstance(event, Mapping) else getattr(event, "type", "")
        if event_type:
            event_types.add(str(event_type))
    if any("died" in event_type or event_type == "death" for event_type in event_types):
        return "death"
    delta = reward_delta(reward)
    # Death dominates any simultaneous positive reward (e.g. kill + death).
    if delta.get("death", 0) > 0:
        return "death"
    if any(value > 0 for value in delta.values()):
        if delta.get("quest", 0) > 0:
            return "success"
        return "progress"
    text = str(feedback or "").casefold()
    if any(token in text for token in ("revived at the starting point", "died")):
        return "death"
    if any(token in text for token in ("invalid", "failed", "cannot", "can't", "unable", "not enough")):
        return "no_effect"
    return "no_effect" if text else "unknown"


@dataclass
class InteractionEvent:
    """One action and the environment result observed on the next state."""

    step: int
    action: str
    outcome: str
    reward_delta: dict[str, float] = field(default_factory=dict)
    state_summary: str = ""
    next_state_summary: str = ""
    feedback: str = ""
    invalid: bool = False
    event_types: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InteractionEvent":
        fields = {
            "step", "action", "outcome", "reward_delta", "state_summary",
            "next_state_summary", "feedback", "invalid", "event_types", "timestamp",
        }
        return cls(**{key: data[key] for key in fields if key in data})


class EventLog:
    """Append-only episode log with a bounded recent-event view."""

    def __init__(self, max_recent: int = 16) -> None:
        if max_recent < 1:
            raise ValueError("max_recent must be at least 1")
        self.max_recent = int(max_recent)
        self.events: list[InteractionEvent] = []
        self._recent: Deque[InteractionEvent] = deque(maxlen=self.max_recent)

    def reset(self) -> None:
        self.events.clear()
        self._recent.clear()

    def append(self, event: InteractionEvent) -> InteractionEvent:
        self.events.append(event)
        self._recent.append(event)
        return event

    def record(
        self,
        *,
        step: int,
        action: Any,
        reward: Any = None,
        feedback: Any = "",
        invalid: bool = False,
        events: Optional[Iterable[Any]] = None,
        state_summary: str = "",
        next_state_summary: str = "",
        outcome: Optional[str] = None,
    ) -> InteractionEvent:
        raw_events = list(events or ())
        event_types = []
        for event in raw_events:
            if isinstance(event, Mapping):
                event_type = event.get("type")
            else:
                event_type = getattr(event, "type", None)
            if event_type:
                event_types.append(str(event_type))
        resolved = outcome or classify_outcome(reward, feedback, invalid, raw_events)
        if (
            resolved == "unknown"
            and state_summary
            and next_state_summary
            and _strip_step(str(state_summary)) == _strip_step(str(next_state_summary))
        ):
            # No reward, no failure signal, and nothing in the world state
            # changed: the action had no effect.  This catches accepted-but-
            # useless actions (e.g. talking to an NPC with nothing new to say)
            # whose feedback text is empty.
            resolved = "no_effect"
        item = InteractionEvent(
            step=int(step),
            action="" if action is None else str(action).strip(),
            outcome=resolved,
            reward_delta=reward_delta(reward),
            state_summary=str(state_summary or ""),
            next_state_summary=str(next_state_summary or ""),
            feedback=str(feedback or ""),
            invalid=bool(invalid),
            event_types=event_types,
        )
        return self.append(item)

    @property
    def recent(self) -> list[InteractionEvent]:
        return list(self._recent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_recent": self.max_recent,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventLog":
        log = cls(max_recent=int(data.get("max_recent", 16)))
        for item in data.get("events", []) or []:
            log.append(InteractionEvent.from_dict(item))
        return log
