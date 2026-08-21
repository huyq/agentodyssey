"""Minimal tool-use agent (harness_v2).

Coding-agent style: the model is the only decision maker.  The harness
provides exactly one tool — ``search_memory`` — and executes it on request;
the model decides whether and when to call it.  There is no ledger, no
event log, no action mask, and no fixed decision protocol.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from agents.llm_agent_config import LLMAgentConfig
from harness_v2.memory import KeywordMemory

TOOL_NAME = "search_memory"


@dataclass
class MemoryToolConfig(LLMAgentConfig):
    max_tool_calls: int = 4
    memory_top_k: int = 3

    system_prompt: str = """
You are the player in a text adventure game. The world is described in text form.
At each turn, you may choose ONE action from the action space below.

Action space:
{}

You have one tool: search_memory. It searches your past experiences (previous observations and the actions you took) by keyword. When recent steps are already shown in the prompt, it only searches older steps beyond them.
- To call it, return: {{"tool": "search_memory", "query": "<space-separated keywords>"}}
- You will then receive the matching past steps, and may call it again or decide your action.
- You may call it at most __MAX_TOOL_CALLS__ times per turn. Use it when you need to recall past locations, objects, NPCs, or the outcome of a past action.

When you are ready to act in the game, return:
{{"reasoning": "A few sentences explaining why you choose the action.", "action": "<action>"}}

Rules:
- Return a single JSON object as the ONLY content of your reply (no extra text before/after).
- Return either one tool call or one action, never both.
- The action must exactly match one option from the action space.
"""

    def __post_init__(self) -> None:
        self.system_prompt = self.system_prompt.replace(
            "__MAX_TOOL_CALLS__", str(self.max_tool_calls)
        )
        super().__post_init__()


def _render_tool_result(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return f'[Memory results for "{query}"]: no matches.'
    lines = [f'[Memory results for "{query}"]:']
    lines.extend(entry["text"] for entry in results)
    return "\n\n".join(lines)


class MemoryToolAgent:
    def __init__(self, id: str, name: str, cfg: Optional[MemoryToolConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = MemoryToolConfig(available_actions=self.available_actions)

        self.memory = KeywordMemory()
        self.llm = self.cfg.get_llm()
        self.memory_paths = ["memory.json"]
        self.last_transcript: List[str] = []
        self._step_counter = 0

    def memorize(self, step: int, info: str) -> None:
        self.memory.add(step, info)

    def _stm_size(self) -> int:
        if not getattr(self.cfg, "enable_short_term_memory", False):
            return 0
        return max(0, int(getattr(self.cfg, "short_term_memory_size", 5)))

    def _render_recent_block(self) -> Optional[str]:
        size = self._stm_size()
        if size <= 0:
            return None
        recent = self.memory.entries[-size:]
        if not recent:
            return None
        parts = ["[Recent steps]"]
        parts.extend(entry["text"] for entry in recent)
        parts.append("[End of recent steps. Call search_memory to recall anything older.]")
        return "\n\n".join(parts)

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs["text"]
        step = obs.get("step")
        if step is None:
            step = self._step_counter
        self._step_counter = int(step) + 1

        max_calls = int(self.cfg.max_tool_calls)
        transcript: List[str] = []
        recent_block = self._render_recent_block()
        if recent_block:
            transcript.append(recent_block)
        transcript.append(f"My Current Observation: {obs_text}")
        responses: List[str] = []
        total_in = 0
        total_out = 0
        calls_made = 0
        action: Optional[str] = None
        reasoning = ""

        for _ in range(max_calls + 1):
            user_prompt = "\n\n".join(transcript)
            lm_output = self.llm.generate(
                user_prompt=user_prompt, system_prompt=self.cfg.system_prompt
            )
            total_in += lm_output["num_input_tokens"]
            total_out += lm_output["num_output_tokens"]
            response = lm_output["response"]
            responses.append(response)

            parsed = self.cfg.json_parser(response) or {}
            if parsed.get("tool") == TOOL_NAME:
                if calls_made >= max_calls:
                    break  # budget exhausted; ignore the call and fall back
                query = str(parsed.get("query") or "")
                results = self.memory.search(
                    query, top_k=int(self.cfg.memory_top_k), exclude_last=self._stm_size()
                )
                calls_made += 1
                transcript.append(response.strip())
                transcript.append(_render_tool_result(query, results))
                if calls_made >= max_calls:
                    transcript.append(
                        "You have used all search calls for this turn. "
                        "Reply with an action JSON now."
                    )
                continue
            action_value = parsed.get("action")
            if isinstance(action_value, str) and action_value.strip():
                # sanitize quotes/backticks some backbones wrap tokens in
                action = action_value.strip().replace('"', "").replace("`", "")
                reasoning = str(parsed.get("reasoning") or "")
            break

        if action is None:
            action = "wait"
            reasoning = reasoning or "Failed to decide; falling back to wait."

        self.last_transcript = list(transcript)

        readable = self.cfg.format_response({"action": action, "reasoning": reasoning})
        self.memorize(int(step), f"[step {int(step)}] {obs_text}\n{readable}")

        return action, total_in, total_out, "\n\n".join(responses)

    def save_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        payload = {
            "agent_id": self.id,
            "agent_name": self.name,
            "cfg": {
                "llm_name": getattr(self.cfg, "llm_name", None),
                "max_tool_calls": getattr(self.cfg, "max_tool_calls", None),
                "memory_top_k": getattr(self.cfg, "memory_top_k", None),
                "enable_short_term_memory": getattr(self.cfg, "enable_short_term_memory", None),
                "short_term_memory_size": getattr(self.cfg, "short_term_memory_size", None),
            },
            "memory": self.memory.to_dict(),
            "step_counter": self._step_counter,
        }
        self.memory.save(path, payload)

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        data = self.memory.load(path)
        if not data:
            print(f"[MemoryToolAgent] No memory file found at {path}", flush=True)
            return
        cfgd = data.get("cfg", {})
        for k in [
            "llm_name",
            "max_tool_calls",
            "memory_top_k",
            "enable_short_term_memory",
            "short_term_memory_size",
        ]:
            if k in cfgd and cfgd[k] is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, cfgd[k])
        self.memory.load_from_dict(data.get("memory", {}))
        self._step_counter = int(data.get("step_counter", len(self.memory.entries)))

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None):
        return self.llm.generate(user_prompt=user_prompt, system_prompt=system_prompt)


@lru_cache(maxsize=None)
def create_memory_tool_agent(Agent: Type):
    class_name = f"MemoryToolAgent__{Agent.__module__}.{Agent.__name__}"

    return type(
        class_name,
        (MemoryToolAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )
