# 0. 当前可执行范围（2026-08-29 实现口径）

为保证另一端 Agent 可以在本仓库中零补丁执行，当前正式可执行范围固定为：

- **Panel A / Panel D**：`Anonymous Vanilla / Personality Only / Generic Summary / Anonymous Gold Profile / Ours`，定位为本方法内部条件与消融子实验。Summary、Personality 和 Ours 复用每个 probe 各自的 Top-10；任何处理请求都不会合并 24 个 probe 的原始评论。Gold 使用 benchmark 官方 profile；Vanilla 不使用 persona 信息。条件共用 Actor 与 evaluator，使用模型/服务默认解码且不显式设置 seed。
- **Panel B 暂移除**：AMADEUS 的官方数据、检索管线及可固定版本实现未随本项目提供，当前不进入默认实验网格或论文结果声明。
- **Panel C 暂移除**：RoleGPT / RoleLLM、PersonaForge、CoSER 涉及外部代码、专用模型或训练产物，当前无法保证另一端 Agent 零补丁运行，故不进入默认实验网格或结果声明。
- Panel B/C 可在未来获得并锁定官方仓库、模型权重、数据许可和复现实验版本后恢复；恢复前必须新增预检、适配器和端到端测试。

跨方法主实验对应 Panel B/C；当前尚未冻结可零补丁执行的官方外部方法，因此本仓库现阶段只执行 Panel A/D 内部子实验。本节覆盖下文旧版 Panel B/C 设计；下文相应内容仅作为未来扩展背景，不代表当前实现已支持。

---

# 1. 整体实验框架

整个 pipeline 固定成 6 步。

### Step 1. 建立 Character Comment Corpus （这一步是在最开始就全部做完，不计入方法的正式流程，独立作为一项contribution）

后台数据保留真实角色 ID：

```json
{
  "comment_id": "c_10231",
  "character_id": "tbbt_sheldon",
  "character_name": "Sheldon Cooper",
  "work": "The Big Bang Theory",
  "platform": "reddit",
  "thread_id": "...",
  "author_hash": "...",
  "timestamp": "...",
  "raw_text": "...",
  "language": "en"
}
```

这一层**不匿名**，因为：

- 要按角色抓评论；

- 要跟 RoleAgentBench / InCharacter / CharacterEval 对齐；

- 最后 evaluator 必须知道这是谁。

从 Step 2 开始，模型看到的是：

```text
TARGET_07
```

而不是 `Sheldon Cooper`。

---

### Step 2. 用固定的侧写问题检索评论

例如：

> 当既定日程、秩序或规则被打乱时，这个人通常如何反应？在什么情况下会例外？

系统后台：

```text
filter: character_id = tbbt_sheldon
query: disruption of routine / rules / unexpected change / exceptions
```

固定检索：

**text-embedding-3-small 精确向量召回 Top-20 → Cohere-rerank-v4.0-pro 完整重排 → Top-10 comments**

评论向量和固定中英文 probe 查询向量在本地准备阶段一次性构建并冻结。每个 probe 先在
该角色评论中做精确余弦 Top-20 召回，再把 20 条匿名候选完整交给 Cohere 重排；请求不设置
`top_n` 或文档 token 上限，本地取 Top-10。BM25、RRF、HNSW、量化与在线重建均不属于
正式方法。GPU 机器只读取冻结后的五条件输入，完全不运行检索或 profile 构建。

---

### Step 3. 从评论中提取 Cue，而不是直接总结 Profile

例如评论表达的是：

> 他通常对临时改变安排非常恼火，会立即要求恢复原来的安排。

不要直接生成：

```text
low openness
high neuroticism
```

而是：

```json
{
  "cue": "unexpected disruption of routine triggers irritation and corrective behavior",
  "context": "daily routine",
  "cue_type": "behavioral_pattern",
  "directness": "behavior_based",
  "support": ["c102", "c391"],
  "counterevidence": ["c817"],
  "confidence": 0.84
}
```

即严格保持：

**Comment → Cue → Latent Hypothesis**

而不是：

**Comment → Personality Label**。

---

### Step 4. 多领域 Cue 聚合成 Reconstructed Person Model

汇总成匿名、结构化且可审计的人物模型，不设置长度目标、截断或归一化：

```text
[Stable Tendencies]
[Motives & Values]
[World/Appraisal Model]
[Affect & Coping]
[Interpersonal Patterns]
[Self & Narrative Themes]
[Situation–Behavior Signatures]
[Expressive Signature]
[Contradictions / Boundary Conditions]
[Unknown / Contested]
```

这才是最终用于实例化的对象。

---

### Step 5. Role Agent 身份盲角色扮演

Role Agent 只得到：

```text
Reconstructed Person Model
+
Benchmark Query
```

不能得到：

- 角色名；

- 作品名；

- 官方 profile；

- wiki；

- 原始角色介绍；

- 原始评论。

---

### Step 6. 后台恢复 character_id，跑原 benchmark evaluator

例如模型不知道自己是 Sheldon：

```json
{
  "anonymous_id": "TARGET_07",
  "character_id": "tbbt_sheldon",
  "benchmark": "RoleAgentBench",
  "task": "general_response",
  "output": "...",
  "score": 0.78
}
```

这样比较的仍然是：

> reconstructed Sheldon vs benchmark gold Sheldon

而不是构造一个无法评价的匿名角色。

---

# 2. 四个实验 Panel

| Panel                     | 数据/角色                  | 主要目的              | 主要方法对比                                                                  |
| ------------------------- | ---------------------- | ----------------- | ----------------------------------------------------------------------- |
| **A. Internal Conditions** | RoleAgentBench Core-10 | 英文内部条件与消融子实验      | Summary / Personality / Gold Profile / Ours / RoleAgent reference |
| **B. CharacterRAG**       | CharacterRAG-15        | 与最新 RAG 型角色方法正面对比 | AMADEUS vs Ours                                                         |
| **C. RoleBench**          | RoleBench-8            | 与经典及最新角色实例化方法对比   | RoleGPT / PersonaForge / CoSER / Ours                                   |
| **D. Chinese Validation** | CharacterEval-6        | 中文、跨语言和日常人物验证     | Profile / Summary / Personality / Ours + CharacterRM                    |

---

# 3. Panel A：英文内部条件与消融子实验

使用 RoleAgentBench 中两个完整作品的 10 个角色：

**Harry Potter**

- Harry Potter

- Hermione Granger

- Ron Weasley

- Draco Malfoy

- Minerva McGonagall

**The Big Bang Theory**

- Sheldon Cooper

- Leonard Hofstadter

- Penny

- Raj Koothrappali

- Howard Wolowitz

RoleAgentBench 官方确实包含这两个 script 及上述角色。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/5875aca1ef70285a35940afbbce0f9fb-Paper-Datasets_and_Benchmarks_Track.pdf?utm_source=chatgpt.com "C  Details on RoleAgentBench"))

这一组的优势是同时包含：

- Harry / Sheldon：高度 iconic；

- Hermione / Malfoy / Penny：中等显著；

- Leonard / Raj / Howard：相对日常、没有那么脸谱化。

因此可以额外分析：

> **Profiling 是否对 subtle / everyday characters 比 iconic characters 更有价值？**

### 评论来源

英文统一：

**PDB + Reddit**

CharacterRAG 动漫角色另外补：

**MyAnimeList / 动漫社区讨论**

每角色样本量按清理后的实际数量报告，不再设置 500 条硬门槛；数据完整性要求为非空、
至少 2 个平台和 100 个独立作者。正式语料只保留 Stack Exchange comment，并排除所有
长度达到 1,000 Unicode 字符的非合成记录。

---

# 4. 侧写领域：固定成 8 个领域 × 3 个问题

# 

24 个问题已经足够形成一个比较完整但仍可控的 profiler。

| Domain                           | 3 个固定 Profiling Questions                                                |
| -------------------------------- | ------------------------------------------------------------------------ |
| **D1 Stable Tendencies**         | 他通常如何计划、履行责任和应对计划被打乱？；熟人与陌生人面前有何稳定差异？；面对规则、不确定性、新事物和风险通常如何？              |
| **D2 Motives & Goals**           | 他反复追求或保护什么？；最害怕失去/避免什么？；愿意为什么目标承担明显代价？                                   |
| **D3 Values & Moral Priorities** | loyalty 与 rule/fairness 冲突时如何选择？；self-interest 与帮助他人冲突时如何？；有什么明显不可跨越的底线？ |
| **D4 Cognitive/Appraisal Style** | 面对模糊意图首先信任还是怀疑？；如何解释失败、批评和权威？；对人和世界有什么反复出现的假设？                           |
| **D5 Affect & Coping**           | 什么最容易触发愤怒、羞耻、焦虑或快乐？；真实情绪和外在表达是否一致？；通常如何调节和恢复？                            |
| **D6 Interpersonal Pattern**     | 面对亲密者、陌生人、竞争者和权威有什么差异？；如何表达关心、拒绝和冲突？；在人际中更偏控制、依赖、支配还是配合？                 |
| **D7 Self & Narrative Identity** | 评论者认为哪些经历真正塑造了他？；他似乎试图维持怎样的自我形象？；评论者反复指出哪些自我盲点或内在矛盾？                     |
| **D8 Situation & Expression**    | 压力、公开挑战、亲近者受威胁时分别如何反应？；哪些情境是一般人格规律的例外？；典型语言、幽默、直接程度和表达方式是什么？             |

这里背后的理论已经够用了：

**Trait / Whole Trait → D1**  
**McAdams → D2/D7**  
**Values → D3**  
**CAPS → D4/D8**  
**Interpersonal → D6**  
**emotion/coping → D5**

没必要在方法里再堆更多理论。

---

# 5. 一个 Probe 怎么运行

例如：

```text
D6-Q2

How does this person typically respond when a close
friend asks for help that conflicts with their own plans?
```

本地准备阶段只在该角色评论库中搜索：

```text
text-embedding-3-small frozen exact-vector Top-20
→ Cohere-rerank-v4.0-pro reranks all 20 candidates
→ local Top-10 comments
```

然后传给 Cue Extractor。

不直接产生：

```text
Agreeableness = high
```

而产生：

```json
{
  "cue": "Frequently accepts personal inconvenience to assist close others.",
  "context": "close relationship",
  "type": "behavioral_pattern",
  "support": ["c102", "c391", "c817"],
  "counterevidence": ["c944"],
  "confidence": 0.82
}
```

只有存在足够独立证据才标记：

```text
SUPPORTED
```

否则：

```text
WEAK / CONTESTED / UNKNOWN
```

**不允许补全评论中没有的信息。**

---

# 6. Cue → Person Model

24 个 probe 完成后，构建匿名 Person Model；保留模型自然生成的内容，不设置长度目标或截断：

```text
[Stable Tendencies]

[Motives and Goals]

[Value Priorities]

[Cognitive / Appraisal Model]

[Affective Dynamics]

[Interpersonal Patterns]

[Self and Narrative Themes]

[Situation–Behavior Signatures]

[Expressive Signature]

[Contradictions and Boundary Conditions]

[Unknown / Contested]
```

最重要的是不要只产生 trait，而要形成：

```text
General tendency:
Strong preference for predictable routines.

IF routine is unexpectedly disrupted
→ irritation → attempt to restore structure.

BUT IF a close other's serious need conflicts with routine
→ may reluctantly prioritize the relationship.

Confidence: High
```

即：

> **trait + context + boundary + counterevidence**

---

# 7. 完整 Data Journey 示例

后台真实人物：

```text
Sheldon Cooper
```

但模型只知道：

```text
TARGET_07
```

### Step 1：侧写问题

```text
When established routines are unexpectedly disrupted,
how does this person normally respond?
When does this pattern change?
```

### Step 2：检索第三方评论

示意：

```text
Comment A:
[TARGET] becomes visibly irritated when people suddenly
change an agreed routine and tends to restore the original plan.

Comment B:
Although [TARGET] complains strongly about disruptions,
he sometimes changes his plans when a close friend genuinely
needs help.
```

### Step 3：Cue extraction

```text
Cue 1:
Unexpected disruption → irritation + corrective behavior
Support: strong

Cue 2:
Close other's serious need → routine may be overridden
Support: moderate
```

### Step 4：重建

```text
[Stable Tendencies]
Strong preference for predictability and explicit structure.

[Interpersonal Pattern]
Close relationships can override procedural preferences.

[Situation–Behavior Signature]
IF ordinary routine disrupted → correct/explain first.
IF close other seriously needs help → reluctantly accommodate.

[Contradiction]
Rigid in routine matters but not absolutely inflexible.
```

### Step 5：实例化

Role Agent 得到：

```text
You are an anonymous individual reconstructed from
independent third-party observations.

Internalize the following person model.
Do not guess your identity.
Do not mechanically display every trait.
Do not invent missing biographical facts.

[PERSON MODEL]
...
```

随后输入 RoleAgentBench 的匿名 General Response / Reaction question。

### Step 6：评价

模型不知道：

```text
TARGET_07 = Sheldon Cooper
```

Evaluator 知道，因此：

```text
TARGET_07
→ tbbt_sheldon
→ Sheldon gold benchmark
→ official evaluator
```

---

# 8. Panel A 的内部条件与消融

这些条件用于分析本方法内部的信息表示与消融，不是跨方法主实验。除 Actor 与 evaluator
保持一致外，各条件保留其原生信息形态，不做长度归一化。

| Method                     | Role Agent 获得什么                            |
| -------------------------- | ------------------------------------------ |
| **Anonymous Vanilla**      | 什么 persona 信息都没有                           |
| **Personality Only**       | 匿名人格描述/Big Five                            |
| **Generic Summary**        | 每个 probe 的 Top-10 分别总结，再聚合为人物总结           |
| **Anonymous Gold Profile** | benchmark 官方 profile，去身份信息，作为 Oracle       |
| **Ours**                   | comments → 24 probes → cues → Person Model |

其中最关键比较是：

> **Ours vs Generic Summary**：证明不是简单总结。

> **Ours vs Personality Only**：证明 person ≠ traits。

> **Ours vs Anonymous Gold Profile**：看社会侧写能恢复多少 oracle persona 信息。

Summary 与 Personality 对 24 个 probe 的 Top-10 分别处理，再只聚合各自的局部输出；Ours
也逐 probe 提取 Cues，再由 Cues 自然构建 Person Model。任何一次 Cue、Summary 或
Personality 局部请求都不得包含其他 probe 的原始评论。Gold 保持 benchmark 或对应方法的
原始 profile。所有条件均不设置 conditioning 长度目标、截断或长度归一化；另行记录真实
token 数并检查模型原生上下文兼容性。

正式评论库使用可复现的确定性清理：删除 Stack Exchange question/answer，并删除
`LENGTH(raw_text) >= 1000` 的非合成记录。评论数量是冻结后按角色报告的描述性样本量，
不通过补抓或降质记录强制达到统一数量。

---

# 9. Panel B：CharacterRAG 方法对比

直接使用 CharacterRAG 官方全部 **15 个角色和 450 QA**。CharacterRAG 提供约 976K 字符的 persona documents；AMADEUS 通过 ACTS、Guided Selection 和 Attribute Extractor 动态提取角色 attributes。([arXiv](https://arxiv.org/abs/2508.02016?utm_source=chatgpt.com "Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations"))

这里两条路线分别是：

### AMADEUS

```text
Official Character Documents
→ RAG
→ Attributes
→ Response
```

### Ours

```text
PDB / Reddit / MAL Comments
→ Profiling
→ Cues
→ Anonymous Person Model
→ Response
```

评价使用官方 450 QA。

这一 Panel 的问题不是：

> “公平条件下谁更强？”

而是：

> **没有角色身份、没有官方人物文档，仅靠第三方社会知觉，能达到 document-grounded CharacterRAG 的多少性能？**

因此 Ours 略低于 AMADEUS 完全可以接受。

[CharacterRAG / AMADEUS paper](https://arxiv.org/abs/2508.02016?utm_source=chatgpt.com)

---

# 10. Panel C：RoleBench 方法对比

选 RoleBench 中评论资源丰富且人格差异明显的 8 人：

- Sheldon Cooper

- Sherlock Holmes

- Gregory House

- Michael Scott

- Peter Parker

- Jack Sparrow

- Tyrion Lannister

- Theodore Twombly

这些角色均在 RoleBench 官方 100-role list 中。([GitHub](https://github.com/InteractiveNLP-Team/RoleLLM-public/blob/main/README.md "RoleLLM-public/README.md at main · InteractiveNLP-Team/RoleLLM-public · GitHub"))

比较：

### RoleGPT / RoleLLM

经典：

```text
Role Profile
+ Character Knowledge
+ Speaking Style
```

RoleBench 有 100 个角色和 168,093 个样本。([GitHub](https://github.com/InteractiveNLP-Team/RoleLLM-public/blob/main/README.md?plain=1&utm_source=chatgpt.com "RoleLLM-public/README.md at main · InteractiveNLP-Team/RoleLLM-public · GitHub"))

### PersonaForge

作为**最新的重要 baseline/reference**。ACL Findings 2026，采用：

```text
Big Five + Defense Mechanisms
+ Speaking Style
+ Dynamic State
+ Dual-process generation
```

并已经在 RoleBench 做 external validation。([ACL Anthology](https://aclanthology.org/2026.findings-acl.386/?utm_source=chatgpt.com "PersonaForge: Psychology-Grounded Dual-Process Architecture for Personality-Consistent Role-Playing Agents - ACL Anthology"))

### CoSER-8B

作为 **specialized RPA reference**，不是公平 baseline，因为经过专门训练。

### Ours

普通 instruction model：

```text
Third-party comments
→ reconstructed anonymous person
```

因此最终不用强调：

> Ours > all SOTA.

而强调：

> **在没有身份、官方 profile 和角色原始语料的情况下，Ours 与 privileged role-playing systems 的 gap 有多大。**

---

# 11. Panel D：中文 CharacterEval

中文必须保留，而且非常适合验证你的核心立意。

CharacterEval 包含 77 个中文小说/剧本角色、1,785 个多轮对话和 13 个细粒度指标，并提供 CharacterRM。([ACL Anthology](https://aclanthology.org/2024.acl-long.638/?utm_source=chatgpt.com "CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation - ACL Anthology"))

选 6 个：

| 角色      | 类型       |
| ------- | -------- |
| **华妃**  | 高度脸谱化    |
| **吕子乔** | 鲜明喜剧人物   |
| **老默**  | 行为/价值冲突型 |
| **朱朝阳** | 复杂、非表面人格 |
| **孟宴臣** | 内敛、关系导向  |
| **许红豆** | 日常、低脸谱化  |

这些角色均真实存在于 CharacterEval 官方 profile 数据中。([GitHub](https://github.com/morecry/CharacterEval/blob/main/data/character_profiles.json "CharacterEval/data/character_profiles.json at main · morecry/CharacterEval · GitHub"))

这组特别好，因为从：

> 华妃

一路到：

> 许红豆

恰好能表现：

**iconic character → ordinary / subtle character**

的连续谱。

### 中文评论来源

主：

**豆瓣 + PDB（转译英文）**

补充：

**虎扑**

同样按清理后的实际评论数报告，并要求至少覆盖 2 个平台和 100 位作者，不设置 500 条
硬门槛。

---

# 12. 中文评价

直接用 CharacterEval 官方 CharacterRM。

主报和你任务最相关的：

**Persona-Behavior (PB)**  
**Persona-Utterance (PU)**  
**Human-Likeness**  
**Conversation Consistency**

CharacterEval 明确把 PB/PU 用于角色行为和说话风格一致性评价，并另外包含 role-playing attractiveness 和 personality back-testing。([OpenReview](https://openreview.net/pdf?id=jTZwpxkcu3&utm_source=chatgpt.com "CharacterEval: A Chinese Benchmark for Role-Playing"))

Knowledge Exposure / Accuracy / Hallucination 可以报告，但**不作为主结论**：

因为 Ours 本身就没有获得百科式角色知识。

中文同样比较：

```text
Anonymous Vanilla
Personality Only
Generic Summary
Anonymous Gold Profile
Ours
```

因此中文和英文内部条件子实验设计是对称的。

---

# 13. 实验模型

不采用 R1、QwQ 之类 reasoning model。

主 Actor 固定为比较成熟的 2024 instruction models：

| Model                        |      |
| ---------------------------- | ---- |
| **Llama-3.1-8B-Instruct**    |      |
| **Qwen2.5-7B-Instruct**      |      |
| **Qwen2.5-14B-Instruct**     |      |
| **Mistral-7B-Instruct-v0.3** | <br> |

Llama-3.1-8B-Instruct 使用 ModelScope 的官方权重镜像
`LLM-Research/Meta-Llama-3.1-8B-Instruct`，固定到提交
`359efdbb8af05b788a4ad4185215c6b8caa9052c`；不从 Hugging Face 下载。

Gemma-2-9B-it 因原生上下文仅为 8K，不进入可执行实验矩阵，也不作替换。
保留 Actor 的最小原生上下文为 32,768 tokens；Llama-3.1-8B-Instruct 为
131,072 tokens。上下文兼容性通过真实 tokenizer 对完整输入静态计数验证，
不截断、不归一化，也不修改模型或 vLLM 默认配置。

Profiler 固定：

> **GPT-5.6 Sol (`gpt-5.6-sol`)**

Profiler 只在有外网的本地准备端通过 `.env` 调用。生成的五条件 prepared 目录经校验和冻结后交给无外网 GPU 端；GPU 端不持有 GPT 密钥，也不构建或修改 profile。

这样同一个 reconstructed person model 再交给不同 Actor，避免把“Profiler 强弱”和“Role-playing 强弱”混起来。

统一：

```text
不显式设置 temperature、top_p 或 seed，全部条件使用模型/vLLM/服务默认解码。
3 次独立 replicate（只记录重复编号，不向服务传 seed）
```

---

# 14. Ablation 只做三个

这三个足够。

### A. Profiling Coverage

```text
25% / 50% / 75% / 100%
```

回答：

> **需要知道一个人多少信息，才能把他实例化出来？**

---

### B. Breadth vs Depth

固定 evidence 条数与来源集合；不控制或归一化生成长度：

```text
Broad:
8 domains × 少量 cues

Narrow:
2 domains × 大量 cues

Balanced:
4–6 domains × 中等 cues
```

回答：

> **认识一个人，是多角度重要，还是某几个维度看得足够深重要？**

---

### C. Identity Leakage

给独立模型 reconstructed Person Model：

```text
Which of these 10/15 characters is this person?
```

记录：

最后最好画：

**X：Identity Recoverability ↓**

**Y：Role Fidelity ↑**

理想结果是：

> **高 Fidelity + 低 Identity Recognition**

也就是：

> 模型表现得像这个人，并不是因为它重新猜出了这个人是谁。

---

# 15. 最终实验规模

这样已经足够，不需要继续膨胀：

### Panel A — 内部条件与消融子实验

**10 characters × 5 actors × 5 conditions**

RoleAgentBench：

**General Response + Reaction**

其中 InCharacter 有 gold 的角色额外跑 BFI fidelity。InCharacter 覆盖 32 个角色和 14 套心理量表。([ACL Anthology](https://aclanthology.org/2024.acl-long.102/?utm_source=chatgpt.com "InCharacter: Evaluating Personality Fidelity in Role-Playing Agents through Psychological Interviews - ACL Anthology"))

### Panel B

**CharacterRAG 15 roles × 450 QA**

重点：

**AMADEUS vs Ours**

### Panel C

**RoleBench 8 roles**

重点：

**RoleGPT / PersonaForge / CoSER / Ours**

只跑 1–2 个 backbone 即可。

### Panel D

**CharacterEval 6 Chinese roles × 3 actors × 5 conditions**

重点：

**PB / PU / Human-Likeness / Consistency**

---

# 17. 最终论文实际上只需要回答四个问题

**RQ1 — Feasibility**  
第三方评论是否足以在身份未知情况下重建一个可用于角色实例化的 Person Model？

**RQ2 — Reconstruction**  
理论驱动的 cue-based profiling 是否优于 Generic Summary 和直接 Personality Description？

**RQ3 — Generalization**  
重建出的 Person Model 能否在评论没有直接描述过的新场景中表现出目标人物的行为、人格和表达规律？

**RQ4 — Information Requirement**  
侧写所需信息量、信息广度以及身份信息，对实例化效果分别有什么影响？

这样整篇实验就很干净：

> **A Panel 证明方法有效；B/C Panel 说明和已有 SOTA/经典路线是什么关系；D Panel 证明不是英语和西方 IP 特例。**

而 CharacterRAG、RoleAgent、RoleLLM、PersonaForge、CoSER、RolePlayBench、PersonaArena、Psy-CoT 基本也把 **2024–2026 这一波最相关的角色实例化路线**覆盖得比较完整了。
