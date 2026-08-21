"""Four-gate experience filter for scaffold-based test-time training.

Model-free, deterministic logic. Every training candidate must be:

1. epistemically verified  — derived from harness events (outcome / reward),
   never from raw observation prose or unverified model text;
2. parameterization-worthy — indexed by an abstract condition cluster, not by
   entity-bound one-off events;
3. informative             — kept only where policy and verified evidence
   disagree (corrections, first verified successes); redundant successes go
   to the replay reservoir;
4. routable                — negatives are never trained alone: a failure only
   becomes a lesson when paired with a cluster-verified positive action.

See TTCL_TEST_TIME_TRAINING_PLAN_V2.md section 4.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

OUTCOME_NEGATIVE = {"invalid", "no_effect", "death"}
OUTCOME_POSITIVE = {"success", "progress"}

# Feedback that indicates the action itself is fine but a prerequisite is
# missing (locked path, missing ingredients, no weapon, ...).  Such failures
# must not poison the action's reputation: 'enter hall' failing because the
# door is locked does not mean 'enter hall' is a bad action — it means the
# agent needs a key first.
PRECONDITION_MARKERS = (
    "locked",
    "not enough",
    "enough",
    "no attack power",
    "no inventory",
    "cannot find",
    "requires",
)


def _is_precondition_failure(feedback: Any) -> bool:
    text = str(feedback or "").casefold()
    return any(marker in text for marker in PRECONDITION_MARKERS)

ConditionKey = Tuple[str, str]


def norm_key(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def build_condition_key(ledger: Optional[Dict[str, Any]]) -> ConditionKey:
    """Abstract condition index for recurrence counting.

    Harness-side bookkeeping only: the key never enters training text, so
    using stage/place names here does not parameterize facts.
    """
    ledger = ledger or {}
    goal = ledger.get("goal") or {}
    location = ledger.get("location") or {}
    stage = str(goal.get("main_stage") or "unknown")
    place = str(location.get("place") or "unknown")
    return (stage, place)


def scaffold_target(action: str, assessment: str, recall: str,
                    candidates: List[str], reasoning: str) -> str:
    """A training target in the exact decision-time response format."""
    return json.dumps({
        "assessment": assessment,
        "recall": recall,
        "candidates": list(candidates),
        "reasoning": reasoning,
        "action": action,
    }, ensure_ascii=False)


def positive_target(action: str) -> str:
    return scaffold_target(
        action=action,
        assessment="This choice led to verified progress in this situation.",
        recall="No contradicting past failures in similar situations.",
        candidates=[f"{action} — verified effective in this situation"],
        reasoning="Repeat the verified behavior in similar situations.",
    )


def correction_target(verified_action: str, failed_action: str) -> str:
    """Negative evidence is only ever trained paired with the verified fix."""
    return scaffold_target(
        action=verified_action,
        assessment="This situation previously blocked progress.",
        recall=(f"Verified: '{failed_action}' failed in this situation; "
                f"'{verified_action}' made verified progress."),
        candidates=[
            f"{verified_action} — verified effective in this situation",
            f"{failed_action} — verified ineffective in this situation",
        ],
        reasoning="Choose the verified effective action over the verified failure.",
    )


@dataclass
class DecisionRecord:
    """One scaffolded decision, recorded at act time, verified one step later."""
    step: int
    condition_key: ConditionKey
    prompt_text: str
    response_text: Optional[str]  # raw model response if it parsed, else None
    action: str


@dataclass
class Lesson:
    kind: str  # "positive" | "correction"
    condition_key: ConditionKey
    prompt_text: str
    target_text: str
    weight: float
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "condition_key": list(self.condition_key),
            "prompt_text": self.prompt_text,
            "target_text": self.target_text,
            "weight": self.weight,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        return cls(
            kind=str(data["kind"]),
            condition_key=tuple(data.get("condition_key") or ("unknown", "unknown")),
            prompt_text=str(data.get("prompt_text") or ""),
            target_text=str(data.get("target_text") or ""),
            weight=float(data.get("weight", 1.0)),
            step=int(data.get("step", -1)),
        )


@dataclass
class ClusterStats:
    attempts: int = 0
    failure_count: int = 0                                       # every negative outcome (repeats count)
    failures: List[str] = field(default_factory=list)          # unique verbatim failed actions
    positives: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # norm key -> {action, count}
    tried: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # norm key -> {action, count}
    precondition_failures: set = field(default_factory=set)    # norm keys blocked by missing prerequisites

    def to_dict(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "failure_count": self.failure_count,
                "failures": list(self.failures), "positives": self.positives,
                "tried": self.tried,
                "precondition_failures": sorted(self.precondition_failures)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterStats":
        return cls(
            attempts=int(data.get("attempts", 0)),
            failure_count=int(data.get("failure_count", 0)),
            failures=list(data.get("failures") or []),
            positives=dict(data.get("positives") or {}),
            tried=dict(data.get("tried") or {}),
            precondition_failures=set(data.get("precondition_failures") or []),
        )


def novelty_override(action: str, stats: Optional[ClusterStats],
                     valid_actions: List[str], threshold: int) -> Tuple[str, bool]:
    """Structural loop-breaker (v2: exploration is structure, not a learning target).

    If the chosen action has been tried >= threshold times in this cluster
    without ever yielding a verified positive, replace it with the least-tried
    currently valid action.  Deterministic; needs no training signal.
    """
    if stats is None or not valid_actions:
        return action, False
    akey = norm_key(action)
    if akey in stats.positives:
        return action, False
    if akey in stats.precondition_failures:
        # Blocked by a missing prerequisite (e.g. locked door): the action is
        # not ineffective, so the loop-breaker must not steer away from it.
        return action, False
    entry = stats.tried.get(akey)
    if entry is None or int(entry["count"]) < int(threshold):
        return action, False

    def _count(candidate: str) -> int:
        e = stats.tried.get(norm_key(candidate))
        return int(e["count"]) if e else 0

    best = min(valid_actions, key=_count)
    if _count(best) < _count(action):
        return best, True
    return action, False


def _magnitude(outcome: str, reward_delta: Optional[Dict[str, Any]]) -> float:
    reward_delta = reward_delta or {}
    if float(reward_delta.get("quest", 0) or 0) > 0:
        return 2.0
    if outcome == "death":
        return 2.0
    if outcome == "success":
        return 2.0
    if outcome == "progress":
        return 1.0
    return 0.5


class LessonFilter:
    """The four-gate cascade plus replay reservoir and batch assembly."""

    def __init__(self, min_recurrence: int = 2, max_lessons: int = 256,
                 max_cluster_share: float = 0.34) -> None:
        self.min_recurrence = int(min_recurrence)
        self.max_lessons = int(max_lessons)
        self.max_cluster_share = float(max_cluster_share)
        self.pending: List[Lesson] = []     # gate-passed, not yet trained on
        self.reservoir: List[Lesson] = []   # replay-only pool
        self.clusters: Dict[ConditionKey, ClusterStats] = {}
        self.events_seen = 0
        self.lessons_emitted = 0
        self._immediate = False             # high-magnitude lesson arrived

    # -- gates 1-3: per-event ------------------------------------------------

    def observe(self, decision: DecisionRecord, event: Dict[str, Any]) -> Optional[Lesson]:
        """Fold one verified (decision, outcome) pair into cluster statistics.

        Returns the lesson if one passed all gates, else None.
        """
        outcome = str(event.get("outcome") or "unknown")
        reward_delta = event.get("reward_delta") or {}
        key = decision.condition_key
        stats = self.clusters.setdefault(key, ClusterStats())
        stats.attempts += 1
        self.events_seen += 1
        akey = norm_key(decision.action)
        if akey:
            tried = stats.tried.setdefault(akey, {"action": decision.action, "count": 0})
            tried["count"] += 1

        high_magnitude = outcome == "death" or float(reward_delta.get("quest", 0) or 0) > 0
        # Disagreement degree = every negative outcome so far, repeats included:
        # repeating an already-failed action is stronger disagreement evidence.
        prior_failures = stats.failure_count
        lesson: Optional[Lesson] = None

        if outcome in OUTCOME_NEGATIVE:
            precondition_failure = _is_precondition_failure(event.get("feedback"))
            if precondition_failure:
                # Blocked by a missing prerequisite (locked door, no
                # ingredients, ...): the action is not categorically bad, so
                # it must not enter failure statistics or training lessons.
                if akey:
                    stats.precondition_failures.add(akey)
            else:
                stats.failure_count += 1
                if akey and all(norm_key(a) != akey for a in stats.failures):
                    stats.failures.append(decision.action)
                # Pairing rule: a failure becomes training data only when a
                # cluster-verified positive exists to pair it with.
                if stats.positives:
                    best = max(stats.positives.values(), key=lambda p: p["count"])
                    base = 2.0 if outcome == "death" else 1.0
                    lesson = Lesson(
                        kind="correction",
                        condition_key=key,
                        prompt_text=decision.prompt_text,
                        target_text=correction_target(best["action"], decision.action),
                        weight=base * (1.0 + 0.5 * min(prior_failures, 4)),
                        step=int(event.get("step", -1)),
                    )
                    if outcome == "death":
                        self._immediate = True

        elif outcome in OUTCOME_POSITIVE:
            is_novel_positive = akey not in stats.positives
            if is_novel_positive:
                stats.positives[akey] = {"action": decision.action, "count": 1}
            else:
                stats.positives[akey]["count"] += 1

            if is_novel_positive and prior_failures > 0:
                # The correction moment: verified success after failures.
                target = decision.response_text or positive_target(decision.action)
                lesson = Lesson(
                    kind="correction",
                    condition_key=key,
                    prompt_text=decision.prompt_text,
                    target_text=target,
                    weight=_magnitude(outcome, reward_delta) * (1.0 + 0.5 * min(prior_failures, 4)),
                    step=int(event.get("step", -1)),
                )
                self._immediate |= high_magnitude
            elif is_novel_positive:
                # First verified success in this cluster (reward-verified).
                target = decision.response_text or positive_target(decision.action)
                lesson = Lesson(
                    kind="positive",
                    condition_key=key,
                    prompt_text=decision.prompt_text,
                    target_text=target,
                    weight=_magnitude(outcome, reward_delta),
                    step=int(event.get("step", -1)),
                )
                self._immediate |= high_magnitude
            else:
                # Redundant success: useful for replay stability, not for
                # driving updates (self-reinforcement only narrows entropy).
                self._remember(self.reservoir, Lesson(
                    kind="positive",
                    condition_key=key,
                    prompt_text=decision.prompt_text,
                    target_text=decision.response_text or positive_target(decision.action),
                    weight=_magnitude(outcome, reward_delta),
                    step=int(event.get("step", -1)),
                ))

        if lesson is not None:
            self._remember(self.pending, lesson)
            self.lessons_emitted += 1
        return lesson

    # -- batch assembly: trigger, throttle, replay mix ------------------------

    def pop_update_batch(self, min_pending: int, replay_per_update: int,
                         force: bool = False) -> Optional[Dict[str, Any]]:
        """Return {"train", "validation", "num_new"} or None if not triggered."""
        if not force and not self._immediate and len(self.pending) < int(min_pending):
            return None

        new_lessons = list(self.pending)
        self.pending.clear()
        self._immediate = False

        # Cluster share throttle: no cluster may dominate one update.
        cap = max(1, math.ceil(len(new_lessons) * self.max_cluster_share))
        counts: Dict[ConditionKey, int] = {}
        kept: List[Lesson] = []
        for lesson in sorted(new_lessons, key=lambda l: l.weight, reverse=True):
            count = counts.get(lesson.condition_key, 0)
            if count < cap:
                kept.append(lesson)
                counts[lesson.condition_key] = count + 1
            else:
                self._remember(self.reservoir, lesson)  # excess stays usable as replay

        replay = list(self.reservoir[-int(replay_per_update):])
        if len(self.reservoir) > int(replay_per_update):
            validation = list(self.reservoir[-2 * int(replay_per_update):-int(replay_per_update)])
        else:
            validation = list(kept)  # weakest fallback: pre/post NLL on the batch itself

        return {"train": kept + replay, "validation": validation, "num_new": len(kept)}

    # -- state ----------------------------------------------------------------

    def _remember(self, pool: List[Lesson], lesson: Lesson) -> None:
        pool.append(lesson)
        if len(pool) > self.max_lessons:
            del pool[: len(pool) - self.max_lessons]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_recurrence": self.min_recurrence,
            "max_lessons": self.max_lessons,
            "max_cluster_share": self.max_cluster_share,
            "pending": [l.to_dict() for l in self.pending],
            "reservoir": [l.to_dict() for l in self.reservoir],
            "clusters": {json.dumps(list(k), ensure_ascii=False): v.to_dict()
                         for k, v in self.clusters.items()},
            "events_seen": self.events_seen,
            "lessons_emitted": self.lessons_emitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonFilter":
        filt = cls(
            min_recurrence=int(data.get("min_recurrence", 2)),
            max_lessons=int(data.get("max_lessons", 256)),
            max_cluster_share=float(data.get("max_cluster_share", 0.34)),
        )
        filt.pending = [Lesson.from_dict(x) for x in data.get("pending") or []]
        filt.reservoir = [Lesson.from_dict(x) for x in data.get("reservoir") or []]
        for key_str, stats in (data.get("clusters") or {}).items():
            try:
                key = tuple(json.loads(key_str))
            except Exception:
                continue
            filt.clusters[key] = ClusterStats.from_dict(stats)
        filt.events_seen = int(data.get("events_seen", 0))
        filt.lessons_emitted = int(data.get("lessons_emitted", 0))
        return filt
