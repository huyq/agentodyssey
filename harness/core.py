"""High-level harness facade for a single AgentOdyssey episode."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable, Mapping, Optional

from .actions import ActionValidityMask, ActionValidation, candidate_actions
from .events import EventLog, reward_delta
from .ledger import StateLedger


def _feedback_for(feedback: Any, agent_id: str) -> str:
    if isinstance(feedback, Mapping):
        return str(feedback.get(agent_id, "") or "")
    return str(feedback or "")


def _reward_for(reward: Any, agent_id: str) -> Any:
    if isinstance(reward, Mapping):
        return reward.get(agent_id, {})
    return reward


def _info_invalid(info: Any, agent_id: str) -> bool:
    if not isinstance(info, Mapping):
        return False
    invalid = info.get("step_invalid_action", info.get("invalid_action", False))
    if isinstance(invalid, Mapping):
        return bool(invalid.get(agent_id, False))
    return bool(invalid)


class Harness:
    """Maintain ledger, recent events, and action masks for one or more agents.

    ``Harness`` is intentionally environment-agnostic.  A caller passes the
    exact observation returned by the environment and, after stepping, the
    reward/feedback/info returned by that same environment.
    """

    def __init__(
        self,
        agent_ids: Optional[Iterable[str]] = None,
        max_recent_events: int = 16,
        known_verbs: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> None:
        """``known_verbs`` maps agent id -> the game's full verb list; when
        supplied, the action mask accepts any candidate the game's verb space
        can parse (matching the environment's own validity semantics)."""
        self.max_recent_events = int(max_recent_events)
        self.known_verbs: dict[str, list[str]] = {
            str(aid): [str(v) for v in verbs]
            for aid, verbs in (known_verbs or {}).items()
        }
        self.ledgers: dict[str, StateLedger] = {}
        self.event_logs: dict[str, EventLog] = {}
        self.action_masks: dict[str, ActionValidityMask] = {}
        for agent_id in agent_ids or ():
            self._ensure(str(agent_id))

    def _ensure(self, agent_id: str) -> tuple[StateLedger, EventLog, ActionValidityMask]:
        if agent_id not in self.ledgers:
            self.ledgers[agent_id] = StateLedger()
            self.event_logs[agent_id] = EventLog(self.max_recent_events)
            self.action_masks[agent_id] = ActionValidityMask(
                known_verbs=self.known_verbs.get(str(agent_id), [])
            )
        return self.ledgers[agent_id], self.event_logs[agent_id], self.action_masks[agent_id]

    def reset(self) -> None:
        """Clear all facts and events at a game boundary."""
        for ledger in self.ledgers.values():
            ledger.reset()
        for log in self.event_logs.values():
            log.reset()

    def observe(self, agent_id: str, observation: Mapping[str, Any] | str) -> dict[str, Any]:
        """Update a ledger from an observation and return an enriched copy."""
        ledger, _, _ = self._ensure(str(agent_id))
        if isinstance(observation, Mapping):
            result = copy.deepcopy(dict(observation))
            text = result.get("text", "")
            step = result.get("step")
            valid_actions = result.get("valid_actions", [])
        else:
            result = {"text": str(observation or "")}
            text, step, valid_actions = result["text"], None, []
        ledger.update_observation(text, step)
        # The env's valid-action list is only available when the official
        # --enable_obs_valid_actions flag is on.  Otherwise derive candidates
        # from the ledger (pure observation content) so the mask and agents
        # still get a contextual candidate pool.
        if not valid_actions:
            valid_actions = candidate_actions(ledger.to_dict())
        result["harness"] = {
            "ledger": ledger.to_dict(),
            "recent_events": [event.to_dict() for event in self.event_logs[str(agent_id)].recent],
            "valid_actions": self.action_masks[str(agent_id)].actions(valid_actions),
        }
        return result

    def validate_action(self, agent_id: str, candidate: Any, observation: Optional[Mapping[str, Any]] = None) -> ActionValidation:
        _, _, mask = self._ensure(str(agent_id))
        valid_actions = observation.get("valid_actions", []) if isinstance(observation, Mapping) else []
        return mask.validate(candidate, valid_actions)

    def after_step(
        self,
        agent_id: str,
        *,
        step: int,
        action: Any,
        previous_observation: Optional[Mapping[str, Any] | str] = None,
        next_observation: Optional[Mapping[str, Any] | str] = None,
        reward: Any = None,
        feedback: Any = "",
        info: Any = None,
        events: Optional[Iterable[Any]] = None,
    ) -> dict[str, Any]:
        ledger, log, _ = self._ensure(str(agent_id))
        previous_summary = ledger.summary()
        if next_observation is not None:
            enriched = self.observe(str(agent_id), next_observation)
            next_summary = ledger.summary()
        else:
            enriched, next_summary = None, previous_summary
        reward_value = _reward_for(reward, str(agent_id))
        feedback_value = _feedback_for(feedback, str(agent_id))
        invalid = _info_invalid(info, str(agent_id))
        event_list = list(events or ())
        event = log.record(
            step=step,
            action=action,
            reward=reward_value,
            feedback=feedback_value,
            invalid=invalid,
            events=event_list,
            state_summary=previous_summary,
            next_state_summary=next_summary,
        )
        ledger.update_progress(reward_value)
        if reward_delta(reward_value).get("quest", 0) > 0:
            ledger.record_fact(
                "main quest stage advanced",
                confidence=1.0,
                step=step,
                source="reward",
            )
        for raw_event in event_list:
            event_type = (
                raw_event.get("type") if isinstance(raw_event, Mapping)
                else getattr(raw_event, "type", None)
            )
            if not event_type:
                continue
            event_type = str(event_type)
            if event_type == "quest_stage_advanced":
                ledger.record_fact("main quest stage advanced", confidence=1.0, step=step, source="environment_event")
            elif event_type == "main_quest_complete":
                ledger.record_fact("main quest completed", confidence=1.0, step=step, source="environment_event")
            elif event_type == "side_quest_completed":
                ledger.record_fact("side quest completed", confidence=1.0, step=step, source="environment_event")
        if event.outcome in {"invalid", "no_effect", "death"}:
            ledger.record_failure(action, feedback_value, step)
        if enriched is not None:
            env_actions = enriched.get("valid_actions", [])
            if not env_actions:
                env_actions = candidate_actions(ledger.to_dict())
            enriched["harness"] = {
                "ledger": ledger.to_dict(),
                "recent_events": [item.to_dict() for item in log.recent],
                "valid_actions": self.action_masks[str(agent_id)].actions(env_actions),
            }
        return {"event": event, "observation": enriched}

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_recent_events": self.max_recent_events,
            "known_verbs": self.known_verbs,
            "agents": {
                agent_id: {
                    "ledger": self.ledgers[agent_id].to_dict(),
                    "events": self.event_logs[agent_id].to_dict(),
                }
                for agent_id in self.ledgers
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Harness":
        harness = cls(
            max_recent_events=int(data.get("max_recent_events", 16)),
            known_verbs=data.get("known_verbs", {}),
        )
        for agent_id, item in (data.get("agents", {}) or {}).items():
            ledger, log, _ = harness._ensure(str(agent_id))
            harness.ledgers[str(agent_id)] = StateLedger.from_dict(item.get("ledger", {}))
            harness.event_logs[str(agent_id)] = EventLog.from_dict(item.get("events", {}))
            harness.action_masks[str(agent_id)] = ActionValidityMask(
                known_verbs=harness.known_verbs.get(str(agent_id), [])
            )
        return harness

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str) -> "Harness":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
