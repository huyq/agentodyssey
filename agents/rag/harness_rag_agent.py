"""RAG agent that consumes the episode-local harness state.

Same retrieval/memory machinery as ``VanillaRAGAgent``; the only difference
is that when the observation carries a ``harness`` field (enabled via
``--enable_harness``), the structured ledger and recent interaction events
are rendered into the user prompt as a compact context block.  Memory
storage and retrieval queries remain identical to the vanilla RAG agent so
the harness contribution can be isolated.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Type

from agents.rag.rag_agent_config import RAGAgentConfig
from agents.rag.vanilla_rag_agent import VanillaRAGAgent


def _render_harness_context(harness: Optional[Dict[str, Any]], max_events: int = 8) -> str:
    """Render the harness ledger + recent events into a compact prompt block."""
    if not harness:
        return ""
    ledger = harness.get("ledger") or {}
    lines: list[str] = ["[Harness State]"]

    location = ledger.get("location") or {}
    if location.get("place") or location.get("area"):
        place = location.get("place") or ""
        area = location.get("area") or ""
        lines.append(f"location={place}, {area}".rstrip(" ,"))
    if location.get("neighbors"):
        lines.append("neighbors=" + ", ".join(location["neighbors"]))

    inventory = ledger.get("inventory") or {}
    items = inventory.get("items") or {}
    if items:
        lines.append(
            "inventory="
            + ", ".join(
                f"{name} x{count}" if count != 1 else name
                for name, count in sorted(items.items())
            )
        )
    if inventory.get("equipped"):
        lines.append("equipped=" + ", ".join(inventory["equipped"]))

    vitals = ledger.get("vitals") or {}
    if vitals:
        lines.append(
            "vitals="
            + ", ".join(f"{key}={value}" for key, value in sorted(vitals.items()))
        )

    goal = ledger.get("goal") or {}
    if goal.get("main_quest"):
        lines.append(f"main_quest={goal['main_quest']}")
    if goal.get("main_stage") and goal.get("main_stage") != "unknown":
        lines.append(f"main_stage={goal['main_stage']}")

    progress = ledger.get("progress") or {}
    if progress:
        lines.append(
            "progress="
            + ", ".join(f"{key}+{value}" for key, value in sorted(progress.items()))
        )

    facts = ledger.get("facts") or []
    if facts:
        lines.append(
            "facts="
            + "; ".join(str(fact.get("claim", "")) for fact in facts[-5:] if fact.get("claim"))
        )

    failures = ledger.get("recent_failures") or []
    if failures:
        lines.append(
            "recent_failures="
            + "; ".join(
                f"step {f.get('step')}: {f.get('action')}"
                for f in failures[-3:]
                if f.get("action")
            )
        )

    # NOTE: deliberately NOT rendering harness['valid_actions'] here.  The env
    # list is only a *subset* of the legal action space (it omits e.g. equip,
    # inspect, and game-specific verbs such as 'invoke law'), so presenting it
    # as the set of available actions misleads the model into avoiding legal
    # verbs.  The agent's system prompt already carries the full action space;
    # the harness mask (with known_verbs) enforces env-valid semantics.

    events = (harness.get("recent_events") or [])[-max_events:]
    if events:
        lines.append("[Harness Recent Events]")
        for event in events:
            delta = event.get("reward_delta") or {}
            delta_str = ", ".join(
                f"{key}+{value}" for key, value in sorted(delta.items()) if value
            )
            line = (
                f"step {event.get('step')}: action='{event.get('action')}'"
                f" -> {event.get('outcome')}"
            )
            if delta_str:
                line += f" ({delta_str})"
            feedback = event.get("feedback")
            if feedback:
                line += f" feedback='{str(feedback)[:80]}'"
            lines.append(line)

    return "\n".join(lines)


class HarnessRAGAgent(VanillaRAGAgent):
    """Vanilla RAG plus harness context injected at prompt time."""

    def __init__(self, id: str, name: str, cfg: Optional[RAGAgentConfig] = None):
        super().__init__(id, name, cfg)

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs["text"]
        harness_context = _render_harness_context(obs.get("harness") or {})
        prompt_obs = f"{harness_context}\n\n{obs_text}" if harness_context else obs_text

        # Identical retrieval / memorization pipeline to VanillaRAGAgent; only
        # the text handed to the LLM prompt differs.
        retrieved_long_term = self.memory.retrieve(
            obs_text, memory_retrieve_limit=self.cfg.memory_retrieve_limit
        )

        if getattr(self.cfg, "enable_short_term_memory", False):
            retrieved_short_term = (
                self.short_term_memory[-self.short_term_memory_size:]
                if self.short_term_memory
                else []
            )
        else:
            retrieved_short_term = []

        short_term_set = set(retrieved_short_term)
        retrieved_long_term_deduped = [
            item for item in retrieved_long_term if item not in short_term_set
        ]

        retrieved = retrieved_long_term_deduped + retrieved_short_term

        current_reflection: Optional[str] = None
        if self.cfg.enable_reflection and retrieved:
            retrieved_text = "\n".join(retrieved)
            current_reflection = self.cfg.reflect(self.llm, obs_text, retrieved_text)

        user_prompt = self.cfg.construct_user_prompt_with_current_reflection(
            retrieved_memories=retrieved,
            observation=prompt_obs,
            current_reflection=current_reflection,
        )
        self.last_user_prompt = user_prompt

        lm_output = self.llm.generate(
            user_prompt=user_prompt, system_prompt=self.cfg.system_prompt
        )

        parsed = self.cfg.response_parser(lm_output["response"])
        parsed_action = parsed["action"]
        readable = self.cfg.format_response(parsed)

        to_summarize = f"{obs_text}\n{readable}"
        step_text = to_summarize

        if self.cfg.enable_summarization:
            summary = self.cfg.summarize(self.llm, to_summarize)
            step_text = summary if summary else to_summarize

        step_memory = self._format_step_memory(
            state_action=step_text,
            current_reflection=current_reflection,
        )
        self.memorize(step_memory)

        return (
            parsed_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )


@lru_cache(maxsize=None)
def create_harness_rag_agent(Agent: Type):
    class_name = f"HarnessRAGAgent__{Agent.__module__}.{Agent.__name__}"

    return type(
        class_name,
        (HarnessRAGAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )
