# AgentOdyssey TTCL Test-Time Training 方案

本文面向 AgentOdyssey/TTCL Challenge，目标是在不把当前游戏事实硬写进模型参数的前提下，让 agent 在 500 步交互中学习可复用的高层策略、技能和失败修正。

## 1. 先明确比赛边界

根据比赛页面和仓库的官方 evaluator，计分运行必须满足：

- 排名游戏是 `remnant`、`mark`、`metropolis`；
- 每个游戏是 standalone run，进入下一个游戏前必须重置 agent memory 和 learned state；
- `max_steps = 500`；
- 计分 backbone 只能是 Qwen3-4B 或 Qwen3.5-4B；
- 不允许在任何 AgentOdyssey 生成的游戏上预训练；
- 每个游戏多次运行时必须报告均值，不能只挑最好的一次；
- 正式排名首先看三局主线 quest reward 的均值，补充 reward 只作为 tie-breaker。

因此，下面的方案都假设：所有训练样本只来自当前这一次游戏的 observation、agent 自己的 action、环境反馈和已获得的 reward；不读取完整 world JSON，也不在三局之间共享 adapter、replay buffer 或技能库。

## 2. 对初始想法的修正

“事实交给 harness，高层语义交给模型”这个方向是对的，但需要把“高层语义”进一步限定为**决策规律**，而不是泛化的语言知识。

适合放在 harness 的内容：

- 当前区域、邻接区域、可见物品和 NPC；
- 背包、装备、生命值、经验、时间和当前任务反馈；
- 已执行动作及其结果；
- 已发现的物品用途、配方、交易对象、门锁和路径；
- 当前主线/支线进度和已经完成的事件；
- 对事实的置信度、来源步数和最近一次验证结果。

适合让模型或可训练模块学习的内容：

- 在什么状态下应该先探索、收集、制作、战斗还是推进任务；
- 哪些前置条件通常需要先完成；
- 某类失败意味着什么，以及下一次应如何修正；
- 如何把长任务拆成短期子目标；
- 哪些动作序列是可重复的技能/option；
- 在信息不完整时如何选择探索动作，而不是反复执行同一个动作。

不建议直接训练的内容：

- 原始 observation 的逐字复述；
- 当前游戏特有的实体名称与静态描述；
- 没有结果验证的模型 reasoning；
- 每一步都把整段 ReAct 输出当作“正确答案”的自训练数据。

当前仓库的 `LoRASFTAgent` 正好存在最后一个风险：它把 observation 和模型生成的 reasoning/action 拼成训练文本，并且默认每步或每五步直接训练，未按 reward 或结果筛选，也没有明确的 response-only loss。这容易强化错误动作、重复动作和模型自己的幻觉。

## 3. 推荐的总架构：双层记忆 + 选择性 TTT

建议把 agent 拆成四层：

```text
Observation
    |
    v
Harness State Ledger       -- 事实、进度、可验证状态，不更新模型权重
    |
    +--> Candidate/Skill Retrieval
    |
    v
Frozen Qwen actor + small LoRA adapter  -- 学习策略、技能和失败修正
    |
    v
Action -> Environment -> reward/feedback
    |
    +--> Event log -> lesson extraction -> replay buffer -> selective update
```

核心原则：

1. harness 负责把长文本 observation 规范化为稳定的结构化状态；
2. actor 每次仍然输出合法 action，但不把所有事实都塞进参数；
3. 只有经过结果验证、具有新颖性或能解释失败的样本才进入 TTT；
4. 更新使用小步、低秩、带 frozen-base KL 的 adapter，尽量避免 Qwen3-4B 崩溃；
5. 参数学习和事实 ledger 都只在当前单局内存在。

## 4. 方案 A：选择性 LoRA TTT + 结构化事实 ledger（首选）

这是最接近当前仓库、实现风险最低的方案。

### 4.1 Harness ledger

每一步从 observation、上一动作和反馈中更新一个 JSON 状态。建议字段：

```json
{
  "goal": {"main_stage": "unknown", "subgoal": "..."},
  "location": {"area": "...", "neighbors": ["..."]},
  "inventory": {"items": {"...": 1}, "equipped": ["..."]},
  "entities": {"visible_objects": [], "visible_npcs": []},
  "facts": [
    {
      "claim": "...",
      "confidence": 0.8,
      "evidence_step": 37,
      "last_verified_step": 42
    }
  ],
  "recent_failures": [],
  "skills": []
}
```

ledger 只保存当前局信息，并在每局开始时清空。对事实采用 append/update，而不是让模型通过参数记忆。例如“某区域需要钥匙”只有在观察或动作反馈验证后才写入，不能仅凭模型 reasoning 写入。

### 4.2 训练样本不是原始轨迹，而是 lesson

将每次交互先记录为 event：

```json
{
  "state_summary": "abstract state, without copying all entity prose",
  "goal": "current subgoal",
  "candidate_action": "craft 1 wooden_key",
  "outcome": "success | progress | no_effect | invalid | death",
  "reward_delta": {"quest": 0, "exploration": 1, "craft": 0, "defeat": 0},
  "next_state_summary": "..."
}
```

只有以下事件才生成训练 lesson：

- 主线阶段推进；
- 首次探索新区域或发现新机制；
- 首次成功制作/战斗/交易；
- 连续失败后找到修正策略；
- 相同状态下两个候选动作的结果明显不同；
- 发现一个可以跨多个状态复用的动作序列。

lesson 的目标格式建议是：

```text
Situation: <抽象状态，不包含不必要的实体描述>
Goal: <短期目标>
Evidence: <动作及环境结果>
Lesson: <可迁移的决策规则>
Next action: <当前状态下的合法动作>
```

例如：

```text
Situation: A locked transition blocks the next quest area and the agent has no key.
Goal: Make progress toward the next main quest stage.
Evidence: Repeated enter attempts failed; a nearby workbench and required materials are visible.
Lesson: When a required path is locked, inspect available recipes and gather the key ingredients before retrying travel.
Next action: inspect inventory
```

这里的 `Lesson` 学的是“遇到类似约束时怎么规划”，而不是把某个游戏的区域名字写进参数。

### 4.3 训练目标

建议不要复用当前实现的“整段文本全部作为 label”的简单目标，而使用：

- response-only causal LM loss：只对 `Lesson`、`Next action` 或 action decision 部分计算 loss；
- 结果权重：成功推进和高价值失败修正权重大于无效动作；
- frozen-base KL：限制 adapter 输出偏离原始 Qwen3 的幅度；
- replay mix：每次更新混入最近的成功 lesson、失败修正 lesson 和少量旧 lesson；
- update cadence：至少积累 4--8 条新 lesson，或出现明确的 progress event 后再更新，不要每一步更新。

一个实用的 loss 形式是：

```text
L = L_action_or_lesson
  + beta * KL(pi_adapter || pi_base)
  + gamma * L_contrastive
```

`L_contrastive` 可让“导致失败的候选动作”概率低于同一状态下经过验证的修正动作。初始时 `beta` 应较大，LoRA rank 可从 4 或 8 开始，确认不崩溃后再尝试 16。

### 4.4 与当前代码的落点

- 在 `agents/parametric/lora_sft_agent.py` 中把 `_format_step_memory` 改为 lesson/event 生成，而不是直接保存原始 observation；
- 新增 `lesson_buffer.py`，负责优先级 replay、去重和按结果加权；
- 在 `TrainableLM.train_on_texts` 中支持 response-only labels、梯度累积和 base-model KL；
- `save_memory/load_memory` 只保存当前局的 ledger、replay 和 adapter；
- 初始实验保留当前 `LoRASFTAgent` 作为 baseline，避免把提升归因到错误位置。

## 5. 方案 B：冻结 actor，在线学习一个轻量 value/reranker

如果直接更新 Qwen3 的 LoRA 仍然不稳定，优先采用这个更保守的方案。

### 5.1 思路

让 Qwen3 负责提出少量合法候选动作，让一个很小的可训练模块负责估计候选动作的长期价值：

```text
Qwen3 -> 3--8 个候选动作
Harness -> 过滤格式非法、明显不可行的动作
Value/Reranker -> 评分并选择动作
Environment -> reward/feedback
TD update -> 更新 value/reranker
```

可训练模块可以是：

- Qwen hidden state 上的线性 value head；
- 一个小型 MLP；
- 或只训练 Qwen LoRA 的 value/ranking prompt 分支。

actor 主体保持 frozen，只有 value/reranker 快速在线更新。这样可以学习“探索还是推进”“先收集还是先战斗”这类高层选择，同时不会因为一次错误 reasoning 破坏语言能力。

### 5.2 训练数据

每条数据是：

```text
(abstract_state, candidate_action, reward_delta, next_abstract_state)
```

使用 TD(0) 或 n-step return：

```text
y_t = r_t + lambda * V(s_{t+1})
```

奖励应优先使用官方可计分的事件增量：主线阶段推进、探索新区域、独特制作和独特击败。对于纯失败/无效动作，给予小负值即可，避免把稀疏主线 reward 过度稀释。

### 5.3 优点和限制

优点：参数少、更新快、灾难性遗忘风险低，适合 500 步短局。

限制：如果 Qwen3 不能提出包含正确动作的候选集，reranker 无法凭空创造技能。因此必须配合探索候选、action validity mask 和结构化 ledger。

这是最适合作为第一版 TTT 对照的方案：先验证“在线价值估计是否能提高 quest reward”，再决定是否需要更新 actor。

## 6. 方案 C：可验证的 skill/option 学习

AgentOdyssey 的很多目标是长链条动作，例如探索区域、拿材料、制作物品、解锁路径、战斗。可以把已经验证过的动作序列压缩成 skill card，让模型学习何时调用 skill，而不是每一步重新规划。

### 6.1 Skill card 格式

```json
{
  "name": "prepare_locked_transition",
  "preconditions": ["path appears locked", "key absent", "materials visible or known"],
  "steps": ["inspect", "collect ingredients", "craft key", "enter target area"],
  "success_condition": "target area entered or quest stage advanced",
  "failure_condition": "ingredient unavailable, combat interruption, death",
  "evidence": [12, 18, 21, 24],
  "confidence": 0.75
}
```

只有在一条动作序列导致明确进度，或者同一序列在不同状态重复成功后，才升级为 skill。失败序列只能进入“待修正 skill”，不能直接成为正样本。

### 6.2 学习方式

- harness 负责检测前置条件和成功条件；
- Qwen3 负责从当前状态选择 skill 或 primitive action；
- LoRA 只训练 skill selection 和 skill completion 的 decision examples；
- skill 内部的实体事实仍由 ledger 提供；
- 每次 skill 成功后提高置信度，失败后记录失败条件并降低置信度。

这样得到的是当前局内可复用的 option policy。它比纯 SFT 更贴合 continual learning，因为学习对象是“一个新发现的可执行技能”，而不是死记某一步的文字。

## 7. 方案 D：短期 episodic memory + 慢速语义 consolidation

可以把 test-time learning 分成两个时间尺度：

### 快速通道

每一步更新 harness ledger 和最近 8--16 条 episodic event，用于马上决策。不更新模型参数。

### 慢速通道

每 25--50 步，或者完成一个主线阶段后，从 episodic buffer 中抽取：

- 重复出现的失败模式；
- 已验证的子目标分解；
- 新发现的技能；
- 对探索/战斗/制作策略的稳定偏好。

将这些 consolidation lesson 做一次小型 LoRA update。每次 update 同时混入旧 lesson replay 和 frozen-base KL。

这个时间尺度分离可以避免当前 SFT 实现的两个问题：

- 每一步都更新，导致模型追随噪声；
- 只记最近 5 条文本，导致已经验证的高层规律被窗口淘汰。

## 8. 推荐实施顺序

### Phase 0：可测 baseline

固定 Qwen3-4B、相同 seed 和 500 steps，跑以下基线：

1. No Memory；
2. 当前 `LoRASFTAgent`；
3. 结构化 ledger 但不训练；
4. frozen actor + value/reranker。

每局记录：主线 quest、四个 supplementary reward、invalid action rate、action diversity、每步 token、adapter update 次数和 lesson 数量。

### Phase 1：先做 harness ledger

不更新模型参数，只实现：

- 事实抽取与置信度；
- 当前目标/子目标状态机；
- action validity mask；
- 失败计数和重复动作检测；
- 可解释的 event log。

这一步可以单独验证事实记忆是否减少重复探索和无效动作。

### Phase 2：加入选择性 lesson SFT

推荐默认配置：

```text
LoRA rank: 4 或 8
targets: q_proj, k_proj, v_proj, o_proj
update: 每 4--8 条 lesson 或每个明确 progress event
replay: 最近成功 lesson + 失败修正 lesson + 少量旧 lesson
loss: response-only + base KL
max lesson length: 512--1024 tokens
```

先只训练 action decision，不训练原始 observation 的 token。只有当这个版本稳定后，才加入抽象化后的 lesson 文本。

### Phase 3：加入 value/reranker 或 skill cards

如果 Phase 2 的 quest reward 仍然不稳定：

- 先加入 value/reranker，检验高层动作排序是否有效；
- 再加入可验证 skill cards，减少长任务中的重复规划。

不建议一开始同时加入 actor LoRA、value head、skill library、reflection 和 summarization，否则无法判断提升来源，也很难定位 collapse。

## 9. 评估设计

比赛只看三局平均主线 reward，但开发阶段应记录更细的指标：

| 维度 | 指标 | 目的 |
|---|---|---|
| 主目标 | quest reward | 官方排名首要指标 |
| 辅助目标 | side quest / exploration / craft / defeat | 官方 tie-breaker |
| 行为质量 | invalid action rate | 判断 harness 和候选过滤是否有效 |
| 记忆 | world knowledge QA / episodic memory QA | 区分事实记忆和经验记忆 |
| 学习效率 | lesson 数、update 次数、每次 update loss | 判断是否过度训练 |
| 稳定性 | action diversity、重复动作比例 | 检测 adapter collapse |
| 成本 | input/output tokens、decision time | 比较 harness 与长上下文的代价 |

每个版本至少跑三次独立 seed，并用仓库的 `challenge_eval.py` 报告每局均值。不要只看单局最高分。

建议的关键 ablation：

1. raw trajectory SFT vs selective lesson SFT；
2. facts in prompt vs facts in ledger；
3. 每步更新 vs event-triggered 更新；
4. 无 replay vs prioritized replay；
5. 无 KL vs base KL；
6. actor LoRA vs frozen actor + value/reranker；
7. 无 skill cards vs skill cards。

## 10. 最终建议

如果目标是尽快做出有竞争力且可解释的第一版，我建议选择：

> **结构化事实 ledger + action validity mask + 事件触发的 selective LoRA lesson learning + 小型 replay + frozen-base KL。**

它比“每一步把 observation/reasoning/action 原样 SFT”更契合 continual learning，也更直接针对当前仓库已经暴露的风险：Qwen3-4B 能力有限、错误自训练、动作多样性塌缩和灾难性遗忘。

如果 actor LoRA 仍然不稳定，则退回到：

> **结构化 ledger + frozen Qwen actor + 在线 value/reranker + 可验证 skill cards。**

这条路线的参数更新更少，但仍然是真正的 test-time learning：模型在当前局中根据奖励学习哪些高层决策更值得选择，而不是只把事实放入上下文。

## 11. 相关仓库入口

- 比赛约束与计分：[challenge_eval.py](/mnt/nas/huyuanquan/code/agentodyssey/challenge_eval.py)
- 单局环境循环：[eval.py](/mnt/nas/huyuanquan/code/agentodyssey/eval.py)
- 当前 LoRA SFT：[agents/parametric/lora_sft_agent.py](/mnt/nas/huyuanquan/code/agentodyssey/agents/parametric/lora_sft_agent.py)
- SFT 超参数：[agents/parametric/param_agent_config.py](/mnt/nas/huyuanquan/code/agentodyssey/agents/parametric/param_agent_config.py)
- Agent 基础接口：[games/base/agent.py](/mnt/nas/huyuanquan/code/agentodyssey/games/base/agent.py)
- 官方比赛规则：[TTCL Challenge](https://ttcl-agents.github.io/#challenge)
