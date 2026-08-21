# 在极简 `harness_v2` 上做 Test-Time Training

## 0. 结论先行

不建议把 `ScaffoldTTTAgent` 的五阶段协议、ledger、action mask 和手工状态簇迁移到
`harness_v2`。对于只有 `search_memory` 的 coding-agent 风格 agent，更自然的 TTT 对象是
**工具策略本身**：

1. 当前状态下是否应调用 memory；
2. 调用时应查什么；
3. 看到结果后是否以及如何修改动作。

推荐的最小方案是：

> **transition memory + 随机化 tool exploration + outcome-filtered tool-use SFT + replay/KL/rollback**

其中最关键的不是先写 trainer，而是先解决两个冷启动问题：

- 现在的 memory entry 不是完整 transition，工具很难可靠回答“上次做这件事结果如何”；
- 自然调用率只有约 1.6%，只用自然轨迹做 TTT 几乎没有正样本，无法靠 TTT 自己把调用率从零附近拉起来。

因此，第一阶段应先提高工具的**可用信息价值**并引入少量随机化调用探索。TTT 只吸收经环境结果支持的工具轨迹，不训练自由 reasoning，不要求固定决策协议。

---

## 1. 当前实现与实测信号

### 1.1 `harness_v2` 与 RAG+STM 的真实差别

`VanillaRAGAgent + STM` 每一步都会执行以下操作：

```text
current observation
    -> embedding retrieve top-k old memories (必经)
    -> append recent STM (必经)
    -> LLM chooses action
```

`MemoryToolAgent + STM` 则是：

```text
current observation + recent STM
    -> LLM chooses [search_memory OR action]
    -> search_memory 后再次调用 LLM
    -> action
```

所以二者不只是 embedding search 与 keyword search 的差别。`harness_v2` 在旧记忆进入上下文前增加了一个高风险门：模型必须先意识到自己缺信息，然后正确输出 tool JSON，再构造一个能命中 keyword index 的 query。任何一步失败都会退化成无长期记忆 agent。

一般 coding agent 能主动使用工具，并不是因为“把工具写进 system prompt”天然足够，而是因为 backbone 已经从大量 tool-use 轨迹中学到了调用习惯。当前 Qwen3-4B 在这一手写 JSON 协议上没有同等先验。

### 1.2 仓库中现有单次对比的诊断

仓库已有 `output/mark_stm_compare` 的 `mark / seed=42 / 500 steps` 对比。该结果只能作为诊断，不能作为统计结论：

| 指标 | MemoryTool + STM | RAG + STM |
|---|---:|---:|
| main quest | 7 | 7 |
| supplementary total | 10 | 18 |
| invalid actions | 12 | 4 |
| 相邻重复动作 | 38 | 16 |
| unique actions | 132 | 129 |
| 平均 input tokens/step | 2010 | 2510 |
| 平均 output tokens/step | 849 | 595 |

MemoryTool 的工具行为为：

- 500 步共调用 8 次，调用率 **1.6%**；
- 每个发生调用的 step 都只调用一次，`max_tool_calls=4` 没有真正成为约束；
- 8 次中 7 次发生在前 250 步，后半局仅 1 次；
- 8 次调用后的当步 reward 都为 0；
- 一次调用后的动作是 invalid。

这支持“调用策略有问题”，但还不能证明“只要提高调用率就会追平 RAG”。也可能同时存在检索质量低、结果难利用、调用增加反而干扰动作等问题。因此第一组实验必须把**调用频率**和**工具信息质量**拆开。

### 1.3 当前 memory schema 降低了工具价值

`MemoryToolAgent._act()` 在动作执行前后没有 outcome callback。它在返回动作前写入：

```text
[step t] obs_t
Reasoning: ...
Action: a_t
```

`a_t` 的反馈会出现在 `obs_{t+1}`，被写进下一条 entry，并与 `a_{t+1}` 放在一起。结果是：

- 搜索动作 `a_t` 更容易命中“当时为什么做”，不一定命中“做完发生了什么”；
- `exclude_last=STM size` 可能恰好排除携带 outcome 的后一条 entry；
- top-k 只返回命中的 entry，不会自动展开相邻 transition；
- keyword score 以原始长 observation 为单位，常见词和重复状态模板会稀释真正的 action-outcome 关系。

如果工具不能稳定返回 outcome，模型少调用它可能是理性的。直接训练“多调用”会优化调用率指标，却未必优化 reward。

---

## 2. 极简设计原则

### 2.1 保持一个工具，不恢复认知脚手架

推理时仍只有一个公开工具：

```json
{"tool": "search_memory", "query": "..."}
```

不新增 ledger、候选枚举、前置条件检查、固定 CoT 字段或游戏状态解析器。训练组件在 agent 内部观察原生 interaction transcript 和官方接口返回的 observation/reward/invalid；它不向 prompt 暴露额外游戏知识。

### 2.2 区分“事实内容”与“工具控制”

事实仍留在 memory 中；TTT 主要改变控制倾向：

- 在重复、受阻或需要旧信息的上下文中，提高 `search_memory` 相对直接行动的概率；
- 学会输出可执行的 tool-call 格式；
- 学会在检索后利用证据，而不是机械重复检索结果。

这比把地点、NPC、配方和当前背包写进参数更接近 V2 文档中“陈述性知识外存，程序性知识入参”的原则。

### 2.3 训练信号高精度优先

一次“call 后 reward=0”不能说明调用有害；一次“call 后 reward>0”也不一定说明调用有因果贡献。应允许大量 interaction 只进入日志而不进入训练。

训练正样本优先使用：

- 检索非空且带来动作修正；
- 修正后的动作得到可验证正结果，或避开 memory 中已验证的同条件失败；
- 查询命中了最终被动作使用的对象/地点/动作信息。

明确的 no-call 样本优先使用：

- 查询结果为空；
- 结果与当前 STM 完全重复；
- 工具重复同一 query，且第二次没有新增结果。

不要仅因最终动作失败就把 tool call 标成负样本，因为未执行的 no-tool 动作可能更差。

---

## 3. 推荐架构

```text
                         +-------------------------+
obs_t + recent STM ----> | Qwen + episode LoRA     |
                         +-------------------------+
                              |              |
                         direct action     tool call
                              |              |
                              |       transition search
                              |              |
                              |       tool result -> Qwen
                              |              |
                              +------ action_t
                                      |
                                  environment
                                      |
                         obs_{t+1}, reward_t, invalid_t
                                      |
                       finalize transition + score trace
                                      |
                       selective tool-use replay buffer
                                      |
                       event-triggered small LoRA update
```

新增组件只有三个，且都在 agent 侧：

1. `TransitionMemory`：将一次行动及其结果绑定成可检索单元；
2. `ToolTrace`：记录调用前输出、query、命中、调用后输出和环境结果；
3. `ToolTTTTrainer`：筛选高置信轨迹并做小步参数更新。

---

## 4. Phase 0：先让 memory 真正可用

### 4.1 将 entry 改成 completed transition

建议 memory 中的基本单元为：

```json
{
  "step": 37,
  "observation": "...",
  "action": "craft 1 oak_rod",
  "next_observation": "...",
  "reward": {"quest": 0, "craft": 1},
  "invalid": false,
  "text": "[step 37] ... Action: ... Outcome: ..."
}
```

`text` 仍可用于当前 keyword search；结构化字段只为完成 transition、日志分析和训练筛选服务，不需要渲染成额外 scaffold。

实现上可在 `eval.py` 的 `env.step()` 后增加一个可选、通用 callback：

```python
agent.observe_transition(
    previous_observation=...,
    action=...,
    next_observation=...,
    reward=reward[agent.id],
    invalid=...,
)
```

只有实现该方法的 agent 接收 callback，其他 agent 行为不变。这比让 agent 在下一步从 observation 文本猜 reward 更可靠，也不要求启用旧 harness。

### 4.2 搜索结果返回 outcome，并展开邻接窗口

短期内可保留 keyword index，但结果至少应明确显示：

```text
[step 37]
Situation: ...
Action: craft 1 oak_rod
Outcome: invalid / no progress / craft reward +1 / observed feedback ...
```

如果暂时不改 entry schema，最低成本补丁是：命中 step `t` 时一并返回 `t+1` entry。但 completed transition 更干净，也避免 query 命中后一条状态时错误归因。

### 4.3 保留原文，不做重型抽象

这里不建议重建 V2 的手工 `(stage, place)` condition cluster。可以只做以下通用规范化：

- lowercase/tokenization；
- entry ID 与 step；
- query 去重；
- exact token overlap score；
- 可选的 embedding rerank 作为独立 ablation，而不是 scaffold 必选项。

如果 transition memory + 强制调用仍显著差于 RAG，再考虑 hybrid/BM25/embedding。不要把检索器问题交给 actor TTT 掩盖。

### 4.4 先固定 tool protocol、thinking mode 和可训练后端

当前 `MemoryToolAgent` 使用 prompt 中描述的 JSON 协议，并没有把 `search_memory` 作为 OpenAI/Qwen chat template 的原生 `tools` schema 传给模型。coding agent 的常见高工具调用率，一部分来自 native tool-call format 的对齐先验；这应作为一个独立的无 TTT baseline 检查，而不是默认手写 JSON 与 native tool calling 等价。

另一个直接影响 TTT 的问题是：

- 当前 vLLM provider 没有暴露 `think` 参数，现有日志中实际生成了 `<think>...</think>`；
- HuggingFace provider 默认 `think=True`；
- 旧 `ScaffoldTrainer.generate()` 显式尝试 `think=False`；
- 当前 vLLM wrapper 只有远程生成接口，没有 trainer 所需的本地 `.model/.tokenizer`。

因此第一版参数 TTT 应先使用本地 HuggingFace backend，并明确选择 `think=False` 或“保留 think 但全部 mask”。训练样本与 inference 必须使用完全相同的 chat template、thinking mode、system prompt 和 tool-result message 格式。否则 call token 的位置和条件分布不同，response-only loss 即使下降也不代表线上调用率会改变。

若最终必须使用 vLLM，需要另行实现 adapter 热加载/刷新及一致的 tool/thinking 参数；这属于部署层工作，不应与 tool-policy 算法同时调试。

---

## 5. Phase 1：先验证“调用率是因”

在训练前做同 seed 的最小对照：

| 组别 | 调用策略 | 搜索后端 | 目的 |
|---|---|---|---|
| A | model 自主 | 当前 keyword | 当前基线 |
| B | 每步自动一次 | 当前 keyword | keyword memory 的信息上限与干扰程度 |
| C | 随机 10%/20% 强制一次 | 当前 keyword | 估计调用频率的剂量效应 |
| D | 每步自动一次 | embedding RAG | 区分调用门与检索后端 |
| E | model 自主 | transition keyword | 单测 memory schema 修复收益 |
| F | model 自主、native tool schema | transition keyword | 检查手写 JSON 协议是否是调用瓶颈 |

自动/强制调用时，query 可以用两种方式做 ablation：

1. 当前 observation 直接作为 query：测检索器，不测 query policy；
2. 让模型在约束提示下只生成 query：测已有 query 能力。

决策规则：

- 若 B/C 比 A 好，低调用率确实是主要瓶颈，继续做 tool-policy TTT；
- 若 B/C 与 A 相同或更差，先修搜索结果和利用能力；
- 若 D 好而 B 差，主要问题是 keyword backend，不是调用策略；
- 若 E 已明显改善，优先保留极简结构，不急着引入参数更新。

这个阶段很重要。当前只有一局结果，而且 main quest 打平；“低调用率导致效果差”目前是合理假设，不是已建立的因果结论。

---

## 6. Phase 2：用随机化 tool exploration 冷启动 TTT

### 6.1 为什么必须有探索

如果只训练自然 tool call：

```text
低调用先验 -> 几乎没有 tool trace -> 没有梯度 -> 继续低调用
```

这是 bootstrap deadlock。解决它不需要固定五阶段 protocol，只需要训练期间有一个小概率的、游戏无关的信息获取探索。

### 6.2 推荐的探索过程

当模型第一次直接给出动作 `a0` 时，以概率 `epsilon_tool` 进行 shadow/forced tool probe：

1. 保存 `a0`，但暂不执行；
2. 用同一个模型生成一个 `search_memory` query；
3. 执行搜索并获得结果；
4. 再让模型基于结果给出动作 `a1`；
5. 执行 `a1`；
6. 下一步拿到环境结果后，记录 `(a0, query, hits, a1, outcome)`。

初版可以总是执行 `a1`。更严谨的版本在 `a0/a1` 间随机选臂并记录 propensity，以便估计 tool treatment effect；但在单局 500 步的小样本下，优先保持实现简单。

建议让 `epsilon_tool` 前高后低，例如：

```text
steps 0-100:   0.20
steps 101-300: 0.10
steps 301-500: 0.05
```

这只是起始值，必须通过非排名环境或少量确认实验选择。不要把调用率硬设成 RAG 的 100%；主动工具的价值之一就是省 token，目标应是 useful-call rate，而不是 call rate 本身。

### 6.3 ToolTrace

每次自然或探索调用都记录：

```json
{
  "step": 120,
  "source": "natural | epsilon_probe",
  "prompt_before_tool": "...",
  "direct_action_before_probe": "a0 or null",
  "tool_response": {"tool": "search_memory", "query": "..."},
  "result_ids": [12, 44],
  "result_text": "...",
  "action_after_tool": "a1",
  "query_repeated": false,
  "result_empty": false,
  "action_changed": true,
  "reward": {},
  "invalid": false,
  "next_observation": "..."
}
```

必须保存结构化 message transcript，而不只是当前日志里拼接后的 `response`。否则无法稳定重建 training chat template，也无法统计空命中和结果利用率。

---

## 7. 训练样本与目标

### 7.1 三种 lesson，而不是完整 scaffold lesson

#### A. `CALL` lesson：学习何时查

训练输入是调用前的原始 conversation，target 是实际有效的 tool JSON：

```json
{"tool": "search_memory", "query": "oak_rod recipe workbench"}
```

只在以下高置信情况进入训练：

- result 非空；
- `a1 != a0` 或自然 call 后确实使用了检索信息；
- `a1` 获得可验证 progress/reward，或者避开了结果中明确记录的失败动作；
- 没有 invalid/parse failure。

#### B. `NO_CALL` lesson：学习何时别查

只用具有近似反事实证据的样本：

- forced probe 返回空结果；
- 返回结果全部已在 STM；
- 重复 query 没有新增 entry；
- `a1 == a0` 且结果未提供任何新 evidence。

target 可以是原先的 direct action，但建议只对“action-vs-tool 的分流 token”施加较高权重，不对自由 reasoning 做 loss。

#### C. `USE_RESULT` lesson：学习查完以后怎么做

输入包含真实 tool result，target 是 `a1`。仅保留环境验证为正的动作修正，或者“旧动作已验证失败、修正动作不再失败”的配对。

这一类可以晚于 `CALL` lesson 启用。第一版先证明能学会调用，再证明检索后动作质量不会下降。

### 7.2 Loss mask

不要训练 `<think>`、自由 reasoning 和整段 observation。推荐 mask：

| token 区域 | 是否训练 | 理由 |
|---|---|---|
| system/user/tool result | 否 | 上下文，不是 target |
| JSON 结构与 `search_memory` | 是，高权重 | 直接学习工具分流与协议 |
| query payload | 可选，低权重 | 学查询，但可能参数化局内专名 |
| final action | 仅 `USE_RESULT` 正样本 | 防止自蒸馏所有动作 |
| reasoning/think | 否 | 不可验证且易合理化 |

如果最优先保持“事实不入参”，可以先完全 mask query payload：模型已有 8 次自然 query 表明基本 query 生成能力存在，最缺的是决定调用。之后再单独 ablate query loss。

### 7.3 更新规则

可复用旧 Scaffold TTT 中经得起简化的训练机制：

- episode-local LoRA，游戏/run 开始时重置；
- rank 4 或 8，attention projections，小学习率，一次 1 epoch；
- 累积 6-12 条高置信 lesson 再更新，高量级 progress 可提前触发；
- 每批新 lesson 与旧 lesson 约 1:1 replay；
- 每个 trace/step 在一个 update 中最多出现一次；
- update 前保存 adapter，验证失败则回滚。

不建议复用旧 `(stage, place)` cluster share、候选列表与前置条件 lesson。这些依赖旧 scaffold 的手工状态抽象，在这里只有额外 inductive bias。

### 7.4 KL 的位置需要改变

旧方案在完整响应上锚定 frozen base；但 frozen base 恰好具有“很少调用工具”的先验。若在同一个调用决策位置施加强 KL，目标 loss 要提高 call 概率，KL 又把它拉回 direct action，两者直接冲突。

推荐：

- 不在正 `CALL` lesson 的 tool-vs-action 分流 token 上做强 KL；
- 在一组 no-tool action replay prompt 和 tool-result 后动作 prompt 上做 KL，保护原 action 能力；
- 监控 held-out direct-action parse rate、action diversity 和平均长度；
- 若调用率上升但动作能力下降，再考虑把 tool decision adapter 与 action generation adapter 分开，而不是一开始就引入双 adapter。

### 7.5 Update validation

每次更新至少检查：

1. held-out positive `CALL` prompts 上 tool-call NLL 是否下降；
2. held-out `NO_CALL` prompts 上误调用率是否没有显著上升；
3. JSON parse rate 是否不下降；
4. direct-action probe 上 action 分布 KL 是否在预算内；
5. query 不为空、长度不过度增长、重复 query 概率不升高。

验证不过即回滚。不要用训练 batch 自身的 NLL 作为唯一验证，它只能证明 optimizer 工作，不能证明工具策略变好。

---

## 8. 一个最小在线算法

```text
initialize base model, episode LoRA, TransitionMemory, replay buffer

for t in 0..499:
    context = current obs + recent STM
    r0 = model(context)

    if r0 is a natural tool call:
        result = search(r0.query)
        a1 = model(context + r0 + result)
        trace = natural_call_trace(r0, result, a1)
        execute a1

    else if Bernoulli(epsilon_tool(t)):
        a0 = parsed action from r0
        q = constrained_query_generation(context)
        result = search(q)
        a1 = model(context + q + result)
        trace = epsilon_probe_trace(a0, q, result, a1)
        execute a1

    else:
        execute r0.action

    receive obs_next, reward, invalid
    finalize TransitionMemory entry
    finalize trace if present
    classify only high-confidence CALL / NO_CALL / USE_RESULT lessons

    if enough lessons or verified high-magnitude progress:
        batch = new lessons + balanced replay
        snapshot adapter
        masked response-only LoRA update
        validate call calibration + action preservation
        rollback on failure
```

第一版甚至可以只实现 `CALL` 和 `NO_CALL`，冻结所有 final-action target。它回答一个非常清晰的问题：**当前局的可验证经验能否让同一个 4B 模型学会更合适地进入 memory 工具？**

---

## 9. 评测指标

### 9.1 不要只看总 call rate

至少记录：

| 指标 | 含义 |
|---|---|
| natural call rate | 排除 epsilon probe 后，模型自己调用的比例 |
| forced probe rate | 训练数据采集成本 |
| search hit rate | query 能否命中旧记忆 |
| novel hit rate | 命中结果是否不在 STM |
| action revision rate | 检索是否改变动作 |
| verified useful-call precision | 改变后是否有正结果/避开已知失败 |
| empty/redundant call rate | 无价值调用比例 |
| repeat-query rate | 是否陷入工具循环 |
| calls per tool turn | `max_tool_calls` 是否被滥用 |
| invalid/repeat action rate | 工具是否改善实际行为 |
| token/latency per env step | 主动检索是否仍比自动 RAG 经济 |
| updates/rollback/lesson counts | TTT 是否真正发生且稳定 |

自然调用率应按局内时间画曲线。如果 TTT 有效，应该看到 epsilon probe 下降时 natural useful calls 上升，而不是只看到 trainer 强制调用。

### 9.2 核心实验矩阵

优先跑以下六组，使用配对 seed：

1. RAG + STM；
2. MemoryTool + STM，原始 entry，无 TTT；
3. MemoryTool + STM，transition entry，无 TTT；
4. 3 + epsilon tool exploration，但不更新权重；
5. 4 + tool-decision TTT；
6. 5 + query/use-result loss。

归因方式：

- `2 -> 3`：memory schema 的收益；
- `3 -> 4`：调用频率/额外 inference 的收益；
- `4 -> 5`：参数化工具策略的净收益；
- `5 -> 6`：学习 query 和结果利用的额外收益；
- `1 vs 5/6`：主动检索能否以更低 token 成本达到自动 RAG 的 reward。

除了三局官方分数，应报告前/后半局 useful-call precision、reward slope 与 invalid/repeat rate。单看第 500 步总 reward 很难区分“更好的初始策略”和“局内学会了工具”。

---

## 10. 与原 V2 方案的取舍

### 10.1 建议保留

- 只用当前 run 的 observation/action/reward/feedback；
- 参数化程序性行为，不把瞬时事实作为主要训练目标；
- response-only loss，不训练自由 CoT；
- verified outcome filtering；
- 负样本需要近似反事实或修正证据；
- 小步 LoRA、replay、更新后验证与回滚；
- 每局/run reset；
- 与“同等信息、权重不更新”的非参数基线比较。

### 10.2 建议删除

- 固定五阶段 reasoning protocol；
- 每步强制 memory query；
- ledger summary；
- valid-action mask 与候选枚举；
- 手工 `(stage, place)` 状态簇；
- 前置条件 claim/checkpoint；
- skill/option/fast-path 状态机。

### 10.3 建议替换

| V2 元素 | 极简替代 |
|---|---|
| 强制查询 | 随机化 epsilon tool exploration，只在训练数据采集期发生 |
| 状态簇复发门 | 原始 transcript 去重 + tool trace 的可验证 utility |
| 候选-选择诊断 | probe 前动作 `a0` 与检索后动作 `a1` 的自然对照 |
| ledger outcome | completed transition 的 reward/invalid/next observation |
| 全 scaffold SFT | tool-decision/query/use-result 的 token-level masked loss |
| 完整响应 KL | 只在 action-preservation probes 上锚定 base |

---

## 11. 建议的实现顺序

### Milestone 1：可观测性与因果诊断

- 在 `eval.py` 增加可选 `observe_transition()` callback；
- 在 `harness_v2/memory.py` 存 completed transition；
- 在 `harness_v2/agent.py` 保存结构化 tool trace；
- 固定 HuggingFace 训练/推理的 chat template 与 thinking mode；
- 增加 call/hit/empty/revision/repeat-query 指标；
- 跑 Phase 1 的 A-F，无参数更新。

验收标准：能回答“模型不查、查不到、还是查到不会用”三者各占多少。

### Milestone 2：工具调用冷启动

- 加 `epsilon_tool` probe；
- 只产生高置信 `CALL/NO_CALL` lesson；
- 新建 `harness_v2/ttt.py`，做 response-only masked LoRA；
- update 后验证 natural call calibration 和 action parse rate；
- 先不训练 query payload 与 final action。

验收标准：后半局 natural useful-call rate 上升，forced probe rate 可下降，invalid/repeat 不恶化。

### Milestone 3：结果利用与延迟信用

- 加 `USE_RESULT` lesson；
- 对 tool 后动作做 paired correction；
- 若即时 reward 太稀疏，再加入短窗口 delayed credit，但必须做独立 ablation；
- 若 keyword 明显成为上限，再换 BM25/embedding rerank。

验收标准：相对“epsilon probe、权重不更新”仍有稳定提升，否则参数化没有证明自身价值。

---

## 12. 主要风险

1. **把调用率当目标**：最容易得到一个频繁查、但 reward 更差且 token 更贵的 agent。优化目标必须是 useful-call precision 与最终 reward。
2. **自举选择偏差**：只从自然 call 学习会放大模型原有的调用分布；epsilon probe 是必要对照。
3. **错误负样本**：call 后失败不等于 call 导致失败。缺少反事实证据时宁可不训练。
4. **稀疏 reward**：制作中间件、寻找钥匙等动作可能当步 reward 为 0。第一版高精度会漏样本，这是可接受的；不要立刻用自由 reflection 补标签。
5. **KL 抵消学习**：base 的低调用先验正是要改变的部分，不能在分流 token 上强锚定。
6. **query 专名入参**：可通过 mask query payload、低权重或只训练 tool-name/JSON token 控制。
7. **训练开销吞掉工具节省**：需同时报告训练 wall time、额外 generation 次数与 tokens/env-step。
8. **单局样本太少**：整局没有足够高置信 lesson 是合法结果。此时应退化为 transition memory agent，而不是强制用噪声更新。

---

## 13. 最推荐的第一版范围

为了最快判断方向是否成立，第一版只做下面四件事：

1. memory 改成 completed transition；
2. 加 10%-20% 随机 tool probe，保存 `a0 -> query/results -> a1`；
3. 只对高置信 `CALL/NO_CALL` 做 token-masked LoRA，不训练 reasoning/query/action；
4. 与“同样 probe 但 adapter frozen”做配对对照。

这版没有旧 harness 的认知协议，也没有手工状态抽象。它只给模型一个通用的信息获取动作，并让模型在当前局中从可验证结果学习何时使用这个动作。若这一步不能超过 frozen-probe baseline，就没有理由继续增加 query TTT、action TTT 或更复杂的 scaffold。
