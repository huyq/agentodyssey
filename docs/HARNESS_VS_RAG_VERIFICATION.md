# Harness vs RAG：AgentOdyssey 验证报告（含规则合规性结论与修复记录）

> 验证目标：TTCL_TEST_TIME_TRAINING_PLAN.md 中 Phase 1 的 harness 系统（状态 ledger + 事件日志 + 动作有效性掩码，`--enable_harness`）相比 AgentOdyssey 论文中的 RAG 系统（`VanillaRAGAgent`：轨迹向量化存储 + top-k 相似度检索注入 prompt）是否有优势。
>
> 实验条件：Qwen3-4B（vLLM 本地服务），官方三局 remnant / mark / metropolis，`max_steps=500`，seed 42（remnant 另加 seed 7 稳健性复跑）。记忆在每局之间全新重置，符合 TTCL Challenge 规则。
>
> 版本说明：**实验数据**（§2）反映验证时点 harness 的实现状态（掩码按子集列表匹配）；**修复**（§3）是本次根据验证结论与规则合规性分析落地的最新代码状态。当前工作区默认行为已是最新修复后的行为。

## 结论（TL;DR）

1. **按验证时点的实现，harness 相对 RAG 没有优势，且有显著劣势**。其 `ActionValidityMask` 只匹配环境 `get_all_valid_actions()` 的**子集列表**，而环境的真实合法性是"动词可被 `parse_action` 解析"；这导致教程必需的 `equip`/`inspect inventory` 与各游戏自定义动词（`lockpick`、`invoke law`、`file`…）被掩码拒绝，且掩码静默替换为 `wait`、隐藏了"无效动作"反馈：
   - RAG + harness（掩码版）：三局平均主 quest = **0.67**（RAG 基线 **2.00**），remnant/mark 两局 0 分瘫痪（499/500 步为 wait）；
   - RAG + ledger + harness（掩码版）：三局平均主 quest = **0.33**，ledger/事件日志无法抵消掩码的破坏。
2. **把掩码修复为"环境可解析即放行"后**，harness（ledger + 事件 + 正确掩码）与 RAG 基线在主指标上**打平**：三局平均主 quest = **2.00 vs 2.00**；tie-breaker 补充奖励 **6.67 vs 7.67**（主要差在 mark 局击杀数）。remnant seed-7 复跑中 harness 版本补充奖励反而显著更好（**15 vs 9**）。
3. **规则合规性（本轮新增结论）**：环境的内部事件流（`res.events`）**不是官方观察协议的一部分**——`env.step()` 只返回 `(obs, reward, terminated, info)`，事件流只被内部 reward 计算与依赖图追踪器消费。官方规则明确 "Revealing any game knowledge to the agent in the prompt beyond the default action space is **not allowed**"，而事件流里包含 `quest_stage_advanced` 的 chapter/stage 内部编号、`quest_spawn` 的 obj_id/area_id、`dep_tracker_hint` 等观测文本不可见的信息。**因此把事件流传给 harness/agent 不允许用于挑战提交**；harness 所需信号（阶段推进、死亡、支线完成）可用 observation 文本 + reward + feedback 合法推导。
4. **总体判定**：当前 harness 相对论文 RAG 系统**没有可复现的优势**。它带来的是行为画像的改变（探索/制作更多、战斗更少、无效动作率归零）与约 40–60% 的输入 token 成本，而不是可计分的提升。已按验证结论落地三项修复（掩码语义、事件分类、prompt 内容），见 §3。

---

## 1. 实验设置

| 配置 | 说明 |
|---|---|
| **A. VanillaRAGAgent** | 论文 RAG 基线：每步 (observation, action) 存入向量库，按 embedding 相似度检索 top-5 注入 prompt |
| **B. VanillaRAGAgent + `--enable_harness`** | RAG 逻辑不变，仅叠加 harness 的动作掩码（ledger 未被 agent 使用）→ 单独度量掩码效应 |
| **C. HarnessRAGAgent + `--enable_harness`** | harness-aware RAG：把 ledger 摘要 + 最近事件渲染进 prompt，同时受掩码约束 |
| **D. HarnessRAGAgent + 修复掩码**（验证用） | C 的掩码修复版；修复后该行为即当前默认的 C |

- LLM：`/mnt/nas/huyuanquan/models/Qwen3-4B`，vLLM 0.14（PPU 加速），`VLLM_PORT=18088`；
- Embedder：Qwen3-Embedding-0.6B（sentence-transformers）；
- 每配置每局 500 步、seed 42；remnant 另跑 seed 7 的 A/D 对照。

## 2. 结果（验证时点）

### 2.1 官方计分（challenge_eval.py，seed 42）

| 配置 | remnant quest/suppl | mark quest/suppl | metropolis quest/suppl | **三局平均 quest** | 平均 suppl |
|---|---|---|---|---|---|
| A. RAG 基线 | 1 / 7 | 1 / 11 | 4 / 4 | **2.000** | 7.333 |
| B. RAG + harness（现状掩码） | 0 / 0 | 0 / 0 | 2 / 5 | **0.667** | 1.667 |
| C. RAG + ledger + harness（现状掩码） | 0 / 0 | 0 / 0 | 1 / 4 | **0.333** | 1.333 |
| D. RAG + ledger + harness（修复掩码） | 1 / 5 | 1 / 6 | 4 / 4 | **2.000** | 5.000 |

### 2.2 行为指标（seed 42）

| 指标 | A | B | C | D |
|---|---|---|---|---|
| 环境判定无效动作率（remnant/mark/metropolis） | 5.0 / 0.2 / 9.0% | 0 / 0 / 0% | 0 / 0 / 0% | 0 / 0 / 0% |
| 被掩码替换比例 | 0 | 99.8 / 98.6 / 92.2% | 49.4 / 24.8 / 29.0% | 0 |
| 去重动作数（remnant/mark/metropolis） | 98 / 123 / 93 | 2 / 3 / 17 | 5 / 7 / 64 | 72 / 84 / 70 |
| 输入 token（remnant） | 772K | 698K | 909K | 1233K（+60%） |

### 2.3 seed 7 复跑（remnant，A vs D）

| 配置 | quest | expl | craft | side | kill | suppl |
|---|---|---|---|---|---|---|
| A. RAG 基线 | 1 | 6 | 0 | 1 | 2 | 9 |
| D. harness 修复版 | 1 | 7 | 1 | 0 | 7 | **15** |

把 remnant 两次 seed 平均后：A 三局平均 quest 2.000 / suppl 7.667；D 2.000 / suppl 6.667。

### 2.4 关键行为差异

- **进度速度**：D 均更早拿到第一段 quest（remnant：step 175 vs 246；mark：123 vs 136）；metropolis 上 A 在 step 32 推进到 quest=4，D 卡在 quest=2 直到 step 392 才追平（D 的法庭动词使用被"子集 valid_actions"误导而近乎不用，见 §2.5）。
- **策略画像**：A 战斗/法庭动词密集（remnant 66 次 attack、metropolis 85 次 `invoke law`/`file objection`/`appeal`）；D 探索与制作更积极（mark expl=4 vs 2、craft=2 vs 1）。
- **无效动作**：A 在 remnant/metropolis 产生 5%/9% 环境无效动作（靠反馈自行纠正）；D 在掩码兜底下为 0。

### 2.5 现状实现的问题（验证时点，均已修复——见 §3）

1. **【致命】掩码语义与环境不一致**：只匹配 `get_all_valid_actions()` 子集，拒绝 `equip`/`inspect inventory`/自定义动词；并把非法动作静默替换为 `wait`，模型收不到环境反馈 → 无限重复同一错误提议（B 日志中 `equip small_bag_1` 被提 499 次；mark 局 88 步中 85 次被掩码）。
2. **【严重】`get_all_valid_actions` 两个上游 bug**：Take-out 段引用未定义变量 `amount`（空手时 UnboundLocalError 崩溃）；Equip 段用实例 id 查基础字典导致 `equip` 永不出现。存在于 `games/base/env.py` 与所有生成环境。**未修改环境代码**：修复方式是让 harness 不再请求 env 的 valid 列表（§3.1），从根上绕开该路径。
3. **【严重】prompt 暴露子集 `valid_actions` 误导模型**：metropolis 局 D 把"invoke law/file/objection"当作不可用动作，法庭机制几乎不用。
4. **【中等】事件/结果分类层未生效**：eval.py 未传环境事件流；`classify_outcome` 把死亡（death reward>0）归类为 `progress`、remnant D 局 500 事件中 481 个 `unknown`、`recent_failures` 为空（死了 9 次 0 条记录）。
5. **【成本】上下文膨胀**：渲染的 harness 块 1.7–2.2KB，是观测文本的 1–4 倍；输入 token +40–60%。

## 3. 修复记录（本轮落地）

### 3.1 掩码语义 → 按游戏动词空间校验（默认行为，零环境改动）

- `harness/actions.py`：`ActionValidityMask` 新增 `known_verbs`。默认校验规则：候选**可按游戏动词空间解析**（动词前缀 + `shlex` 剩余参数，镜像环境 `parse_action` 的判据）即接受。动词来自各 agent 的 `available_actions`——即系统 prompt 中展示的**默认动作空间**（规则明确允许的部分），不读取任何游戏内部定义。
- `eval.py`：`--enable_harness` **不再隐含** `--enable_obs_valid_actions`（此前是 `args.enable_obs_valid_actions or args.enable_harness`，正是它触发了 §2.5 第 2 条的上游 bug）。默认 harness 运行不请求 env 的 valid 列表；只有用户显式传官方 flag `--enable_obs_valid_actions` 时，列表才作为附加匹配层（命中→返回环境规范拼写；未命中→掩码）。
- **环境代码保持与官方仓库逐字节一致**（`games/` 无任何改动）：提交代码不含环境补丁，组织者核验时可复现。
- 行为验证：remnant 教程局 25 步 0 次掩码，`equip`/`inspect`/`unequip` 全部放行；无 env 列表时不可解析动作标记 `unknown` 放行、由环境原生拒绝并给出可见反馈（不静默吞掉反馈）。

### 3.2 事件分类（协议内信号，默认）

- `harness/events.py`：`classify_outcome` 把 `death` 奖励优先于通用正奖励分支（死亡不再被归类为 `progress`）；feedback 文本含 "revived at the starting point"/"died" 也判为死亡。仅用协议内信号（reward + feedback 文本），**不涉及环境内部事件流**（按 §5 的合规性结论，事件流不接入 harness）。

### 3.3 停止向 prompt 暴露子集 valid_actions（默认）

- `agents/rag/harness_rag_agent.py`：`_render_harness_context` 不再渲染 `valid_actions=` 行（动作空间由系统 prompt 提供；掩码负责合法性）。验证：渲染块中不再出现 `valid_actions`。

### 3.4 环境事件流：不接入

验证中发现 harness 的结果分类层未生效（§2.5 第 4 条），但修复**不涉及**把环境内部事件流（`res.events`）接入 harness：官方规则禁止向 prompt 注入超出默认动作空间的游戏知识，而事件流不是官方观察协议的一部分（`env.step()` 不返回它），且含观测文本不可见的内部信息（chapter/stage 编号、`obj_id`/`area_id`、`dep_tracker_hint`）。因此事件流**不接入**（未修改 `env.step` 返回协议），所需信号全部由 §3.2 的协议内信号（reward / feedback 文本）推导。

### 3.5 其它

- `providers/vllm.py`：探测超时 3s→30s、`max_new_tokens`→`max_tokens`（适配 OpenAI 客户端/vLLM）——运行环境适配，与 harness 无关。
- `tests/test_harness.py`：新增掩码 known_verbs 解析（含/不含 env 列表两种模式）、死亡分类优先级、known_verbs 持久化等 5 个测试，8/8 通过。

## 4. 复现方式

```bash
# 启动 LLM 服务（一次）
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas/huyuanquan/models/Qwen3-4B --served-model-name Qwen/Qwen3-4B \
  --host 127.0.0.1 --port 18088 --max-model-len 32768 --gpu-memory-utilization 0.9

# 每局独立跑（记忆自动重置；--run_dir 各自独立）
VLLM_PORT=18088 .venv/bin/python eval.py --game_name remnant --agent VanillaRAGAgent \
  --llm_provider vllm --llm_name Qwen/Qwen3-4B --max_steps 500 --seed 42 \
  --run_dir output/ttcl_compare/remnant/VanillaRAGAgent
VLLM_PORT=18088 .venv/bin/python eval.py --game_name remnant --agent VanillaRAGAgent \
  --llm_provider vllm --llm_name Qwen/Qwen3-4B --max_steps 500 --seed 42 --enable_harness \
  --run_dir output/ttcl_compare/remnant/VanillaRAGAgent_harness
VLLM_PORT=18088 .venv/bin/python eval.py --game_name remnant --agent HarnessRAGAgent \
  --llm_provider vllm --llm_name Qwen/Qwen3-4B --max_steps 500 --seed 42 --enable_harness \
  --run_dir output/ttcl_compare/remnant/HarnessRAGAgent

# 汇总官方分数
.venv/bin/python challenge_eval.py --remnant <run_dir> --mark <run_dir> --metropolis <run_dir>
```

批量脚本：`scripts/run_game_parallel.sh <game>`（每局 3 配置并行）、`scripts/analyze_harness_compare.py`、`scripts/summarize_harness_compare.py`。

## 5. 规则合规性备忘（供挑战提交核对）

| 信号 | 是否允许 | 说明 |
|---|---|---|
| observation 文本（含 feedback） | ✅ | 官方接口返回值 |
| reward breakdown（RewardBreakdown） | ✅ | 官方接口返回值 |
| `info.step_invalid_action` | ✅ | 官方接口返回值 |
| `valid_actions`（`--enable_obs_valid_actions`） | ✅ | 官方开关，默认动作空间的当前上下文子集 |
| 环境内部事件流 `res.events` | ❌ | 非官方接口；含 chapter/stage 编号、obj_id/area_id、dep_tracker_hint 等观测不可见信息，违反 "Revealing any game knowledge to the agent in the prompt..." |
| world/环境 config JSON | ❌ | 计划文档已明确"不读取完整 world JSON" |
| 跨局共享 memory/adapter/buffer | ❌ | 每局 standalone，必须重置 |
| 在 AgentOdyssey 游戏上预训练 | ❌ | 官方规则明文禁止 |

harness 的默认路径（ledger 从 observation 解析、事实从 reward delta 记录、outcome 从 reward/feedback 分类、掩码按默认动作空间的动词校验）全部落在 ✅ 行内。**环境/游戏代码（`games/`）保持与官方仓库一致，零改动**——提交代码中不应包含任何环境补丁，否则组织者核验与复现时会出问题。

## 6. 对后续工作的建议（对应 TTCL 方案 Phase 1→2）

1. **以修复后的默认行为重新评估 harness 与 RAG 的差距**：本轮 14 局实验是在旧掩码下跑的；修复后 B/C 的掩码不再瘫痪教程，需要重跑三局确认修复是否让 harness（C）追平/超越 A。预期变化：C 的无效动作与重复提议大幅下降，metropolis 的法庭动词可用性恢复。
2. **ledger 本体是可靠的**：位置/邻居/背包/属性解析抽查 100% 正确，进度与 reward 对齐——继续作为 Phase 2 选择性 TTT 的事实来源没有问题。
3. **不要期待 harness 单独带来计分提升**：修复前数据显示主 quest 打平、补充奖励互有胜负；harness 的可验证价值是无效动作归零、更早拿到首段 quest、探索/制作覆盖面更广。拉开差距的方向仍是方案 A/C 的"事件触发的选择性 lesson TTT + replay + frozen-base KL"（Phase 2/3）。
4. **TTT 训练信号只用协议内信号**：lesson 提取基于 observation/reward/feedback（§3.2 已把死亡/进度分类修好），不需要环境内部事件流；事件流不接入 harness。
5. **提交红线**：`games/`（环境与游戏逻辑）保持官方原样；harness 只依赖 agent 侧代码（`harness/`、`agents/`、`eval.py` 的集成）。`--enable_obs_valid_actions` 仅在需要时显式使用（注意官方 env 存在 §2.5 第 2 条的上游 bug，空手状态下会崩溃；harness 默认不触发该路径）。
