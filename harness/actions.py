"""Action candidate normalization and validity masking.

The environment is the source of truth for valid actions.  This module only
normalizes and filters candidates supplied by an agent; it never attempts to
infer hidden world rules from the action text.

Semantics
---------
A candidate is accepted when either:

1. it literally matches an entry of the environment-provided valid-action
   list (canonical spelling is returned), or
2. ``known_verbs`` are supplied and the candidate parses against the game's
   verb space (verb prefix + ``shlex``-parseable remainder), mirroring the
   environment's own ``parse_action``.  This matters because environments
   expose only a *subset* of legal actions (``get_all_valid_actions``), and
   the true legality rule is "the verb is known and the action parses" —
   e.g. ``equip small_bag_1`` or a game-specific ``invoke law ...`` are
   legal but absent from the subset list.

Only candidates failing both checks are masked.  If no valid-action list and
no verbs are available, the candidate is passed through as ``unknown``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _key(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _parseable(candidate: str, known_verbs: Sequence[str]) -> bool:
    """Mirror the environment's ``parse_action``: verb prefix + remainder."""
    s = candidate.strip()
    if not s:
        return False
    s_lower = s.lower()
    verbs = sorted((str(v).lower() for v in known_verbs), key=len, reverse=True)
    for verb in verbs:
        if not verb:
            continue
        if s_lower.startswith(verb):
            if len(s_lower) == len(verb) or s_lower[len(verb)].isspace():
                remainder = s[len(verb):].strip()
                try:
                    shlex.split(remainder)
                except ValueError:
                    return False
                return True
    return False


def normalize_valid_actions(valid_actions: Any) -> list[str]:
    """Return a stable, de-duplicated list from list- or dict-shaped actions.

    Older environments expose a flat list while newer wrappers may group
    actions by verb.  Empty values and non-string entries are discarded.
    """

    if isinstance(valid_actions, Mapping):
        values: list[Any] = []
        for group in valid_actions.values():
            if isinstance(group, (str, bytes)):
                values.append(group)
            elif isinstance(group, Iterable):
                values.extend(group)
    elif isinstance(valid_actions, (str, bytes)):
        values = [valid_actions]
    elif isinstance(valid_actions, Iterable):
        values = list(valid_actions)
    else:
        values = []

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        action = _text(value)
        key = _key(action)
        if action and key not in seen:
            result.append(action)
            seen.add(key)
    return result


@dataclass(frozen=True)
class ActionValidation:
    """Result of applying a validity mask to one candidate action."""

    action: str
    accepted: bool
    reason: str
    valid_actions: tuple[str, ...] = ()


def candidate_actions(ledger: Optional[Mapping[str, Any]]) -> list[str]:
    """Contextual candidate actions derived from the harness ledger.

    The ledger only stores facts already stated by observations, so the
    candidates are a re-statement of the observation protocol (visible
    entities / inventory / neighbors), never world-JSON or hidden game
    knowledge.  Used to populate ``harness["valid_actions"]`` when the
    environment does not expose its own valid-action list.

    The list is advisory: the env's real legality is verb-space parseability,
    so candidates here may still be rejected by the environment (and actions
    outside this list that parse are accepted by the mask).
    """
    if not ledger:
        return []
    out: list[str] = []
    goal = ledger.get("goal") or {}
    required = goal.get("required_action")
    if required:
        out.append(str(required))
    location = ledger.get("location") or {}
    for neighbor in location.get("neighbors") or []:
        out.append(f"enter {neighbor}")
    inventory = ledger.get("inventory") or {}
    for item in inventory.get("items") or {}:
        out.append(f"drop {item}")
        out.append(f"equip {item}")
        out.append(f"unequip {item}")
        out.append(f"inspect {item}")
        out.append(f"store 1 {item} inventory")
        out.append(f"take out {item} inventory")
        out.append(f"discard 1 {item} inventory")
    for item in inventory.get("equipped") or []:
        out.append(f"unequip {item}")
        out.append(f"inspect {item}")
    entities = ledger.get("entities") or {}
    for obj in entities.get("visible_objects") or []:
        out.append(f"pick up {obj}")
        out.append(f"inspect {obj}")
    for npc in entities.get("visible_npcs") or []:
        out.append(f"talk to {npc}")
        out.append(f"attack {npc}")
    out.append("inspect inventory")
    out.append("defend")
    out.append("wait")
    seen: set[str] = set()
    result: list[str] = []
    for action in out:
        key = _key(action)
        if key and key not in seen:
            result.append(action)
            seen.add(key)
    return result


class ActionValidityMask:
    """Filter model actions against environment-provided valid actions.

    Matching is whitespace- and case-insensitive, but the canonical spelling
    from the environment is returned.  When ``known_verbs`` (the game's full
    verb space) is supplied, candidates that parse against it are accepted
    even when absent from the subset list.  If neither a valid-action list
    nor verbs are available, the candidate is passed through and marked
    ``unknown`` so this helper does not silently turn a partially
    instrumented environment into a wait loop.
    """

    def __init__(self, fallback_action: str = "wait", known_verbs: Optional[Sequence[str]] = None) -> None:
        self.fallback_action = _text(fallback_action) or "wait"
        self.known_verbs = [str(v) for v in (known_verbs or [])]

    def actions(self, valid_actions: Any) -> list[str]:
        return normalize_valid_actions(valid_actions)

    def validate(self, candidate: Any, valid_actions: Any, known_verbs: Optional[Sequence[str]] = None) -> ActionValidation:
        candidate_text = _text(candidate)
        actions = self.actions(valid_actions)
        action_tuple = tuple(actions)
        verbs = [str(v) for v in (known_verbs if known_verbs is not None else self.known_verbs)]

        if not candidate_text:
            if not actions:
                return ActionValidation(candidate_text, False, "unknown", action_tuple)
            return ActionValidation(self.fallback_action, False, "masked", action_tuple)

        candidate_key = _key(candidate_text)
        for action in actions:
            if _key(action) == candidate_key:
                return ActionValidation(action, True, "valid", action_tuple)

        # The env's subset list omits many legal verbs (equip, inspect, custom
        # game verbs...).  If the candidate parses against the game's verb
        # space, the environment itself would accept it — pass it through.
        if verbs and _parseable(candidate_text, verbs):
            return ActionValidation(candidate_text, True, "valid", action_tuple)

        if not actions:
            return ActionValidation(candidate_text, bool(candidate_text), "unknown", action_tuple)

        fallback_key = _key(self.fallback_action)
        fallback = next((a for a in actions if _key(a) == fallback_key), actions[0])
        return ActionValidation(fallback, False, "masked", action_tuple)

    def filter(self, candidates: Sequence[Any], valid_actions: Any) -> list[str]:
        """Keep valid candidates in input order, using canonical action text."""

        actions = self.actions(valid_actions)
        if isinstance(candidates, (str, bytes)):
            candidates = [candidates]
        if not actions:
            return [_text(candidate) for candidate in candidates if _text(candidate)]
        by_key = {_key(action): action for action in actions}
        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = _key(candidate)
            if key in by_key and key not in seen:
                result.append(by_key[key])
                seen.add(key)
        return result
