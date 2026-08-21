"""Scaffold TTT agent: cognitive-scaffold prompting + selective parametric learning.

Implements TTCL_TEST_TIME_TRAINING_PLAN_V2.md:

- harness = tools (ledger / events / valid-action mask, consumed from
  ``obs["harness"]`` as produced by ``--enable_harness``) + scaffold (a fixed
  five-stage reasoning protocol enforced by the system prompt);
- only harness-verified, cluster-indexed, disagreement-bearing experience is
  parameterized (the four-gate ``LessonFilter``);
- training uses response-only weighted loss on the exact decision-time chat
  format, a frozen-base KL anchor, replay mixing, and update-level validation
  with rollback;
- outcome feedback arrives one step later inside ``obs["harness"]["recent_events"]``
  (the eval loop calls ``Harness.after_step`` after ``env.step``), so no eval-loop
  interface change is needed.

Deferred from the v2 plan (documented future work): skill/option execution,
familiarity-adaptive fast path, a value/reranker head, and an explicit
calibration lesson type (calibration mismatches currently enter cluster
failure statistics via invalid/no_effect outcomes instead).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Type

import torch
from torch.optim import AdamW

from agents.parametric.param_agent_config import ParamAgentConfig
from agents.parametric.ttt_filter import (
    DecisionRecord,
    Lesson,
    LessonFilter,
    build_condition_key,
    norm_key,
    novelty_override,
    positive_target,
)
from harness.actions import normalize_valid_actions
from utils import atomic_write

from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class ScaffoldTTTConfig(ParamAgentConfig):
    # training schedule (v2: small steps, low rank, event-triggered)
    max_seq_len: int = 2048
    lr: float = 1e-5
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 1
    bf16: bool = True

    # KL anchor to the frozen base
    kl_beta: float = 0.1
    kl_budget: float = 0.5          # mean per-token KL (nats); above -> roll back the update
    rollback_nll_margin: float = 0.05  # tolerate 5% held-out NLL regression

    # four-gate filter
    min_recurrence: int = 2
    max_lessons: int = 256
    disagreements_per_update: int = 6
    replay_per_update: int = 4
    max_cluster_share: float = 0.34

    # structural loop-breaker: override an action tried this many times in the
    # same condition cluster without a verified positive
    repeat_threshold: int = 3

    system_prompt: str = """
You are the player in a text adventure game. The world is described in text form.

You must follow this fixed decision protocol:
1. ASSESS the current goal and main obstacle using the structured state summary (do not restate facts).
2. RECALL what the memory query results imply (verified successes and failures in similar situations).
3. Compare CANDIDATES taken from the valid action list; check each candidate's preconditions against the structured state.
4. DECIDE on one action.

Action space:
{}

Output format (STRICT):
Return a single JSON object with exactly these keys:
{{
  "assessment": "one line: current goal and main obstacle",
  "recall": "one line: what verified past outcomes imply now",
  "candidates": ["<action> — one-line precondition check", "..."],
  "reasoning": "one line: final comparison",
  "action": "<one action copied verbatim from the valid action list>"
}}

Rules:
- The JSON must be the ONLY content in your reply (no extra text before/after).
- Copy "action" verbatim from the valid action list provided in the input.
- Never repeat an action that the memory query results mark as failed in a similar situation.
- If the memory query results show an action was tried many times in the current situation with no new result, choose a different action.
"""

    @property
    def lora_config(self) -> dict:
        # v2 default: rank 8 on attention projections
        return {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }


class ScaffoldTrainer:
    """Response-only, weighted, KL-anchored LoRA training with rollback."""

    def __init__(self, base, cfg: ScaffoldTTTConfig):
        self.base = base
        self.model = base.model
        self.tokenizer = base.tokenizer
        self.device = base.device
        self.cfg = cfg
        self._peft: Optional[PeftModel] = None
        self._opt: Optional[AdamW] = None
        self.total_updates = 0
        self.total_steps = 0
        self.rolled_back = 0

    # -- infrastructure -------------------------------------------------------

    def _ensure_peft(self):
        if self._peft is not None:
            return self._peft
        lc = self.cfg.lora_config
        peft_cfg = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM",
        )
        self._peft = get_peft_model(self.model, peft_cfg)
        self.base.model = self._peft
        return self._peft

    def _optimizer(self) -> AdamW:
        if self._opt is None:
            params = [p for p in self._peft.parameters() if p.requires_grad]
            self._opt = AdamW(params, lr=self.cfg.lr)
        return self._opt

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None):
        # think=False keeps the decision-time format fixed, so training and
        # inference share one tokenization contract.
        try:
            return self.base.generate(user_prompt=user_prompt, system_prompt=system_prompt, think=False)
        except TypeError:
            return self.base.generate(user_prompt=user_prompt, system_prompt=system_prompt)

    # -- encoding --------------------------------------------------------------

    def _encode(self, lesson: Lesson) -> Dict[str, Any]:
        """Chat-templated prompt masked out; only the target JSON earns loss."""
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": lesson.prompt_text},
        ]
        try:
            prefix_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prefix_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prefix_ids = self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(lesson.target_text, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            target_ids = target_ids + [eos]
        budget = self.cfg.max_seq_len - len(target_ids)
        if budget < 8:
            target_ids = target_ids[: self.cfg.max_seq_len // 2]
            budget = self.cfg.max_seq_len - len(target_ids)
        if len(prefix_ids) > budget:
            prefix_ids = prefix_ids[-budget:]  # keep the recent end of the prompt
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": float(lesson.weight),
        }

    def _collate(self, feats: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones_like(f["input_ids"]), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "weights": torch.tensor([f["weight"] for f in feats], dtype=torch.float32),
        }

    @staticmethod
    def _select_target_logits(logits: torch.Tensor, labels: torch.Tensor):
        """(sel_logits[mask,V] float32, flat targets, sample index per position)."""
        logits = logits[:, :-1, :]
        tgt = labels[:, 1:]
        mask = tgt != -100
        if not mask.any():
            return None
        sel = logits[mask].float()
        return sel, tgt[mask], mask.nonzero()[:, 0]

    def _weighted_nll(self, sel, tgt, sample_idx, weights) -> torch.Tensor:
        lp = torch.log_softmax(sel, dim=-1)
        nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        total = sel.new_zeros(())
        wsum = 0.0
        for i, w in enumerate(weights):
            sel_i = sample_idx == i
            if sel_i.any():
                total = total + nll[sel_i].mean() * float(w)
                wsum += float(w)
        return total / max(wsum, 1e-8)

    def _batch_loss(self, batch: Dict[str, Any], with_kl: bool):
        out = self._peft(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        picked = self._select_target_logits(out.logits, batch["labels"])
        if picked is None:
            return None, 0.0
        sel, tgt, sample_idx = picked
        weights = batch["weights"]
        ce = self._weighted_nll(sel, tgt, sample_idx, weights)
        kl_value = 0.0
        if with_kl and self.cfg.kl_beta > 0:
            la = torch.log_softmax(sel, dim=-1)
            lb = None
            with torch.no_grad():
                try:
                    with self._peft.disable_adapter():
                        base_out = self._peft(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    base_picked = self._select_target_logits(base_out.logits, batch["labels"])
                    if base_picked is not None:
                        lb = torch.log_softmax(base_picked[0], dim=-1)
                except (AttributeError, RuntimeError):
                    lb = None
            if lb is not None:
                pa = la.exp()
                kl = (pa * (la - lb)).sum(-1).mean()
                ce = ce + self.cfg.kl_beta * kl
                kl_value = float(kl.detach())
        return ce, kl_value

    # -- validation / rollback --------------------------------------------------

    @torch.no_grad()
    def _nll(self, lessons: List[Lesson]) -> Optional[float]:
        if not lessons or self._peft is None:
            return None
        self._peft.eval()
        total, wsum = 0.0, 0.0
        for i in range(0, len(lessons), self.cfg.batch_size):
            feats = [self._encode(l) for l in lessons[i: i + self.cfg.batch_size]]
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in self._collate(feats).items()}
            autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                          enabled=(self.cfg.bf16 and torch.cuda.is_available()))
            with autocast:
                out = self._peft(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            picked = self._select_target_logits(out.logits, batch["labels"])
            if picked is None:
                continue
            sel, tgt, sample_idx = picked
            lp = torch.log_softmax(sel, dim=-1)
            nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            for j, w in enumerate(batch["weights"].tolist()):
                sel_j = sample_idx == j
                if sel_j.any():
                    total += float(nll[sel_j].mean()) * w
                    wsum += w
        return total / wsum if wsum > 0 else None

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone()
                for k, v in self._peft.state_dict().items() if "lora_" in k}

    def _restore(self, snapshot: Dict[str, torch.Tensor]) -> None:
        state = self._peft.state_dict()
        with torch.no_grad():
            for k, v in snapshot.items():
                state[k].copy_(v.to(self.device))

    # -- training ----------------------------------------------------------------

    def train_on_lessons(self, lessons: List[Lesson], validation: List[Lesson]) -> Dict[str, Any]:
        if not lessons:
            return {"steps": 0, "rolled_back": False}
        peft_m = self._ensure_peft()
        validation = validation or lessons

        pre_nll = self._nll(validation)
        snapshot = self._snapshot()

        feats = [self._encode(l) for l in lessons]
        opt = self._optimizer()
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.cfg.fp16 and not self.cfg.bf16 and torch.cuda.is_available())
        )
        peft_m.train()
        total_steps = 0
        kl_seen: List[float] = []
        rng = torch.Generator().manual_seed(0)
        order = list(range(len(feats)))
        for _ in range(self.cfg.epochs):
            order = torch.randperm(len(feats), generator=rng).tolist()
            for i in range(0, len(order), self.cfg.batch_size):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in self._collate([feats[j] for j in order[i: i + self.cfg.batch_size]]).items()}
                if self.cfg.bf16 and torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss, kl_value = self._batch_loss(batch, with_kl=True)
                else:
                    loss, kl_value = self._batch_loss(batch, with_kl=True)
                if loss is None:
                    continue
                kl_seen.append(kl_value)
                loss = loss / self.cfg.grad_accum
                scaler.scale(loss).backward()
                if (total_steps + 1) % self.cfg.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in peft_m.parameters() if p.requires_grad], max_norm=1.0
                    )
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                total_steps += 1

        post_nll = self._nll(validation)
        mean_kl = sum(kl_seen) / len(kl_seen) if kl_seen else 0.0
        rolled_back = False
        nll_regressed = (
            pre_nll is not None and post_nll is not None
            and post_nll > pre_nll * (1.0 + self.cfg.rollback_nll_margin)
        )
        if nll_regressed or mean_kl > self.cfg.kl_budget:
            self._restore(snapshot)
            rolled_back = True
            self.rolled_back += 1
        else:
            self.total_updates += 1
        self.total_steps += total_steps
        peft_m.eval()
        print(
            f"[ScaffoldTTT] update: {len(lessons)} lessons, {total_steps} steps, "
            f"kl={mean_kl:.4f}, nll {pre_nll} -> {post_nll}, rolled_back={rolled_back}",
            flush=True,
        )
        return {"steps": total_steps, "rolled_back": rolled_back, "kl": mean_kl,
                "pre_nll": pre_nll, "post_nll": post_nll}

    # -- persistence ---------------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        if self._peft is None:
            return
        os.makedirs(path, exist_ok=True)
        self._peft.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        self._peft = PeftModel.from_pretrained(self.model, path, is_trainable=True)
        self.base.model = self._peft
        self._opt = None
        self._peft.eval()


class ScaffoldTTTAgent:
    """Five-stage scaffold protocol + four-gate selective LoRA TTT."""

    def __init__(self, id: str, name: str, cfg: Optional[ScaffoldTTTConfig] = None):
        super().__init__(id, name)
        if cfg:
            cfg.available_actions = self.available_actions
            cfg.__post_init__()
            self.cfg = cfg
        else:
            self.cfg = ScaffoldTTTConfig(available_actions=self.available_actions)

        self.llm = self.cfg.get_llm()
        self.trainable = hasattr(self.llm, "model") and hasattr(self.llm, "tokenizer")
        self.trainer = ScaffoldTrainer(self.llm, self.cfg) if self.trainable else None
        if not self.trainable:
            print("[ScaffoldTTT] LLM has no local weights; running scaffold-only (no TTT).", flush=True)

        self.filter = LessonFilter(
            min_recurrence=self.cfg.min_recurrence,
            max_lessons=self.cfg.max_lessons,
            max_cluster_share=self.cfg.max_cluster_share,
        )
        self._pending_decisions: Deque[DecisionRecord] = deque(maxlen=64)
        self._last_event_step = -1
        self._decision_counter = 0
        self._warned_no_harness = False

        self.memory_paths = ["memory.json"]
        self.adapter_subdir = "lora"

    # -- harness ingestion ----------------------------------------------------

    def _ingest_events(self, events: List[Dict[str, Any]]) -> None:
        for ev in sorted(events, key=lambda e: int(e.get("step", -1))):
            step = int(ev.get("step", -1))
            if step <= self._last_event_step:
                continue
            self._last_event_step = step
            if not self._pending_decisions:
                continue
            decision = self._pending_decisions.popleft()
            lesson = self.filter.observe(decision, ev)
            if lesson is not None:
                print(
                    f"[ScaffoldTTT] lesson ({lesson.kind}, w={lesson.weight:.2f}, "
                    f"cluster={lesson.condition_key}): {decision.action} -> {ev.get('outcome')}",
                    flush=True,
                )

    def _maybe_train(self) -> None:
        if self.trainer is None:
            return
        batch = self.filter.pop_update_batch(
            min_pending=self.cfg.disagreements_per_update,
            replay_per_update=self.cfg.replay_per_update,
        )
        if batch is None:
            return
        self.trainer.train_on_lessons(batch["train"], batch["validation"])

    # -- prompting --------------------------------------------------------------

    def _render_state_block(self, ledger: Dict[str, Any]) -> str:
        goal = ledger.get("goal") or {}
        loc = ledger.get("location") or {}
        inv = ledger.get("inventory") or {}
        vit = ledger.get("vitals") or {}
        facts = ledger.get("facts") or []
        failures = ledger.get("recent_failures") or []
        lines = [
            f"Main quest stage: {goal.get('main_stage', 'unknown')}",
        ]
        if goal.get("required_action"):
            lines.insert(0, f"Required action: {goal['required_action']}")
        if goal.get("main_quest"):
            lines.append(f"Main quest: {goal['main_quest']}")
        lines += [
            f"Location: {loc.get('place', '')}, {loc.get('area', '')}",
            f"Neighboring areas: {', '.join(loc.get('neighbors') or []) or 'none'}",
            f"Inventory: {inv.get('items') or 'empty'}",
            f"Equipped: {', '.join(inv.get('equipped') or []) or 'none'}",
        ]
        if vit:
            lines.append("Vitals: " + ", ".join(f"{k}={v}" for k, v in vit.items()))
        if facts:
            lines.append("Verified facts: " + "; ".join(
                str(f.get("claim", "")) for f in facts[-5:] if f.get("claim")))
        if failures:
            lines.append("Recent failures: " + "; ".join(
                f"{f.get('action', '?')} ({str(f.get('feedback') or '')[:60]})"
                for f in failures[-5:]))
        return "\n".join(lines)

    def _render_cluster_block(self, condition_key) -> str:
        stats = self.filter.clusters.get(condition_key)
        if stats is None or (not stats.failures and not stats.positives and not stats.tried):
            return ""
        parts = []
        if stats.tried:
            top = sorted(stats.tried.values(), key=lambda e: e["count"], reverse=True)[:5]
            parts.append("already tried: " + ", ".join(
                f"'{e['action']}'x{e['count']}" for e in top))
        if stats.precondition_failures:
            parts.append("blocked by missing prerequisites: " + ", ".join(
                f"'{a}'" for a in sorted(stats.precondition_failures)[-5:]))
        if stats.failures:
            parts.append("failed: " + ", ".join(f"'{a}'" for a in stats.failures[-5:]))
        if stats.positives:
            parts.append("verified effective: " + ", ".join(
                f"'{p['action']}'x{p['count']}" for p in stats.positives.values()))
        return "In similar past situations (same stage and area): " + "; ".join(parts)

    def _build_prompt(self, obs_text: str, ledger: Dict[str, Any],
                      valid_actions: List[str], condition_key) -> str:
        blocks = ["[State]\n" + self._render_state_block(ledger)]
        cluster_block = self._render_cluster_block(condition_key)
        memory_lines = ["[Memory query results]"]
        memory_lines.append(cluster_block if cluster_block else
                            "No verified outcomes recorded for similar situations yet.")
        blocks.append("\n".join(memory_lines))
        if valid_actions:
            shown = valid_actions[:40]
            blocks.append("[Valid actions]\n" + "\n".join(f"- {a}" for a in shown))
        blocks.append("[Current observation]\n" + obs_text)
        blocks.append(self.cfg.action_prompt)
        return "\n\n".join(blocks)

    def _resolve_action(self, raw: Any, valid_actions: List[str]) -> str:
        raw_key = norm_key(raw)
        if not valid_actions:
            return str(raw or "").strip() or "wait"
        by_key = {norm_key(a): a for a in valid_actions}
        if raw_key in by_key:
            return by_key[raw_key]
        for a in valid_actions:
            akey = norm_key(a)
            if akey and raw_key and (akey in raw_key or raw_key in akey):
                return a
        # The candidate list is advisory (observation-derived or the env's
        # subset); it omits legal verbs such as game-specific ones (lockpick,
        # invoke law, ...).  The environment's own legality criterion is
        # verb-space parseability — if the raw action parses against the game's
        # verb space, pass it through instead of forcing it into the list.
        if self._parses_verb_space(raw):
            return str(raw).strip()
        return by_key.get("wait", valid_actions[0])

    def _parses_verb_space(self, raw: Any) -> bool:
        from harness.actions import _parseable
        return _parseable(str(raw or ""), [a.verb for a in self.available_actions])

    # -- main loop -----------------------------------------------------------------

    def _act(self, obs: Dict):
        obs_text = obs.get("text", "") if isinstance(obs, dict) else str(obs)
        harness = obs.get("harness") if isinstance(obs, dict) else None
        if harness is None and not self._warned_no_harness:
            print("[ScaffoldTTT] no harness block in observation; "
                  "run with --enable_harness for full functionality.", flush=True)
            self._warned_no_harness = True
        harness = harness or {}
        ledger = harness.get("ledger") or {}
        events = harness.get("recent_events") or []
        valid_actions = normalize_valid_actions(
            harness.get("valid_actions") or (obs.get("valid_actions") if isinstance(obs, dict) else None)
        )

        # Outcomes arrive one step later, embedded in the harness event log.
        self._ingest_events(events)
        self._maybe_train()

        condition_key = build_condition_key(ledger)
        user_prompt = self._build_prompt(obs_text, ledger, valid_actions, condition_key)
        lm_output = self.trainer.generate(user_prompt, self.cfg.system_prompt) \
            if self.trainer is not None else self.llm.generate(user_prompt, self.cfg.system_prompt)

        parsed = self.cfg.json_parser(lm_output["response"]) or {}
        raw_action = parsed.get("action")
        if not isinstance(raw_action, str) or not raw_action.strip():
            raw_action = ""
        action = self._resolve_action(raw_action, valid_actions)

        # Tutorial directive: the observation states a required action.  It is
        # the immediate goal, so it wins over both the model's proposal and the
        # structural loop-breaker (which would otherwise steer away after early
        # locked-door failures).
        required_action = (ledger.get("goal") or {}).get("required_action")
        forced = bool(
            required_action and norm_key(action) != norm_key(required_action)
        )
        if forced:
            print(
                f"[ScaffoldTTT] required action directive; executing "
                f"'{required_action}' instead of '{action}'",
                flush=True,
            )
            action = required_action

        # Structural loop-breaker: replace an action that has been tried
        # repeatedly in this situation without verified effect (v2: exploration
        # is structure, not a learning target).  Never applies to a forced
        # required action.
        cluster_stats = self.filter.clusters.get(condition_key)
        overridden = forced
        if not forced:
            action, overridden = novelty_override(
                action, cluster_stats, valid_actions, self.cfg.repeat_threshold
            )
        if overridden and not forced:
            print(f"[ScaffoldTTT] repeat loop detected; overriding to '{action}'", flush=True)

        # A decision only becomes a training candidate if its response parsed
        # in the protocol format (epistemic gate at the source).  An overridden
        # decision never reuses the model's own trace as a training target:
        # the trace argued for the looped action.
        response_text = lm_output["response"] if (raw_action and not overridden) else None
        self._pending_decisions.append(DecisionRecord(
            step=self._decision_counter,
            condition_key=condition_key,
            prompt_text=user_prompt,
            response_text=response_text,
            action=action,
        ))
        self._decision_counter += 1

        return (
            action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    # -- persistence ----------------------------------------------------------------

    def save_memory(self, full_memory_dir: str) -> None:
        os.makedirs(full_memory_dir, exist_ok=True)
        if self.trainer is not None:
            self.trainer.save_adapter(os.path.join(full_memory_dir, self.adapter_subdir))
        data = {
            "agent_id": self.id,
            "agent_name": self.name,
            "filter": self.filter.to_dict(),
            "counters": {
                "decision_counter": self._decision_counter,
                "last_event_step": self._last_event_step,
                "total_updates": self.trainer.total_updates if self.trainer else 0,
                "total_steps": self.trainer.total_steps if self.trainer else 0,
                "rolled_back": self.trainer.rolled_back if self.trainer else 0,
            },
            "adapter_subdir": self.adapter_subdir,
        }
        atomic_write(
            os.path.join(full_memory_dir, self.memory_paths[0]),
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def load_memory(self, full_memory_dir: str) -> None:
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.filter = LessonFilter.from_dict(data.get("filter") or {})
        counters = data.get("counters") or {}
        self._decision_counter = int(counters.get("decision_counter", 0))
        self._last_event_step = int(counters.get("last_event_step", -1))
        if self.trainer is not None:
            self.trainer.total_updates = int(counters.get("total_updates", 0))
            self.trainer.total_steps = int(counters.get("total_steps", 0))
            self.trainer.rolled_back = int(counters.get("rolled_back", 0))
            adapter_dir = os.path.join(full_memory_dir, data.get("adapter_subdir", self.adapter_subdir))
            if os.path.exists(adapter_dir):
                self.trainer.load_adapter(adapter_dir)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None):
        if self.trainer is not None:
            return self.trainer.generate(user_prompt, system_prompt)
        return self.llm.generate(user_prompt, system_prompt)


@lru_cache(maxsize=None)
def create_scaffold_ttt_agent(Agent: Type):
    class_name = (
        f"ScaffoldTTTAgent__"
        f"{Agent.__module__}.{Agent.__name__}"
    )

    return type(
        class_name,
        (ScaffoldTTTAgent, Agent),
        {
            "__module__": Agent.__module__,
            "__agent__": Agent,
        },
    )
