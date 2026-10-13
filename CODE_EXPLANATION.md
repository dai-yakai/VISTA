# VISTA 项目完整代码讲解

本文档面向 `/home/ubuntu/vista` 项目，目标是把 VISTA-Net++ 的方法论、实现思路、代码结构、关键函数和实验流水线完整讲清楚。文档重点围绕三个问题展开：

1. 用户意图如何被编译成结构化目标。
2. 信念图如何构造，网络状态包含哪些部分，不确定性如何被量化。
3. 系统如何结合历史案例和当前假设进行推理、生成、验证和修复。

需要先明确一个边界：`methodology_v3_network_llm_clean.md` 描述的是完整的 VISTA-Net++ 方法框架，它是一个更一般的网络运维智能体设计；`src/` 下的代码是在 NetArena MALT 基准测试上的工程化落地版本。由于 MALT 任务本身是容量规划图操作任务，当前代码实现会把一些论文级组件简化成可跑通、可评测、可复现实验结果的形式。例如，文档中有显式的信念编码器、信息增益规划器和多世界反事实模拟；代码中对应为信念快照、案例检索、多候选采样、证明置信度、图副本执行和验证门控。

---

## 1. 方法论总览

### 1.1 核心洞察

VISTA 的核心洞察是：网络管理不应该被建模成一次性从提示词到代码或从提示词到命令，而应该被建模成一个围绕“信念状态”的安全决策循环。

传统大语言模型智能体在网络管理里容易出现几个问题：

- 只把当前提示词当上下文，不显式保存网络状态。
- 只生成一个候选动作，没有维护多个可能解释或多个候选执行分支。
- 缺少对输出动作的结构化检查，只看代码是否能运行或答案是否像样。
- 检索历史案例时只做文本相似，而没有把历史过程当成可复用的程序模板和安全先验。
- 执行动作前没有稳定的验证门，尤其没有把语法、类型、语义、图不变量和模拟执行放在同一个证明对象中。

VISTA 的设计用一个统一的状态对象 `B_t` 串起理解、表示、检索、假设、规划和验证：

```text
用户问题 q
  -> 类型化目标 G
  -> 信念状态 B_t
  -> 检索到的历史案例 R_t
  -> 假设集合 H_t / 候选程序集合
  -> 带证明的验证结果 P_t
  -> 修复，或输出最终答案
```

### 1.2 创新点

代码和方法论体现了以下创新点：

- **类型化目标优先**：自然语言先被编译为结构化目标 `G`，包括操作序列、返回形态、不变量、允许动作、停止条件，而不是直接让模型“猜代码”。
- **以信念为中心的状态**：网络图不是隐含在提示词里，而是通过 `BeliefGraph` 查询真实 MALT 图，把目标相关节点的存在性、类型、孩子数量和局部结构写入提示词。
- **案例记忆作为过程先验**：历史案例不是普通检索增强生成文档，而是训练集中已解决的 `(question, answer, task_label, complexity, op)` 程序记录。检索结果被改写为少样本程序模板，作为当前推理的过程先验。
- **多假设采样**：VISTA 不只生成一个程序，而是对同一个 `(G, B_t, R_t)` 采样多个候选，形成工程实现中的 `H_t`。
- **带证明的验证器**：候选程序必须通过语法、函数签名、模拟执行、图不变量、语义等价和安全检查。每个候选都有 `Proof`，不是简单通过/失败。
- **证明引导修复**：失败不是直接结束，而是把 `Proof.repair_hint` 注入修复提示词，让大语言模型根据具体诊断进行贪心修复。
- **对齐基准测试的评分**：最终正确性和安全性仍由 NetArena 官方 `BenchmarkEvaluator` 判定，避免自定义评估偏移。

### 1.3 文档方法和代码方法的对应关系

`methodology_v3_network_llm_clean.md` 里把 VISTA-Net++ 写成六个阶段：

1. 意图编译器。
2. 时序类型化信念图构建器。
3. 信念编码器。
4. 假设生成器。
5. 信息增益规划器。
6. 带证明的验证器和反事实模拟。

当前 MALT 工程实现中的对应关系如下：

| 方法论组件 | 工程实现 | 主要文件 |
|---|---|---|
| 意图编译器 | 正则解析器编译自然语言为类型化目标 | `src/intent.py` |
| 信念图构建器 | 从真实 MALT 图构造目标相关信念快照 | `src/belief.py`, `src/bridge.py` |
| 信念编码器 | 当前未训练图神经网络/时序编码器，压缩为提示词摘要 | `src/belief.py` |
| 案例记忆检索 | BGE 嵌入向量 + FAISS 前 k 个历史案例 | `src/retrieval.py` |
| 假设生成器 | 大语言模型 `n` 次采样得到多个候选程序 | `src/planner.py`, `src/run_vista.py`, `src/run_all_gen.py` |
| 信息增益规划器 | 方法论公式存在；工程上用证明置信度、修复和安全门控近似 | `src/planner.py`, `src/verifier.py` |
| 类型化动作构造 | 生成 `process_graph`，必要时补齐输出格式约束 | `src/prompts.py`, `src/polish.py` |
| 反事实验证 | 在图副本上执行候选程序并检查语义和不变量 | `src/verifier.py` |
| 外部基准测试评分 | NetArena 官方评估器 | `src/bridge.py`, `src/parallel_score.py` |

---

## 2. 端到端执行流程

### 2.1 VISTA 单条查询的逻辑流程

以 `src/planner.py::run_vista_one()` 为主线，一条查询的处理过程是：

```text
用户问题
  -> compile_intent(question)
       得到类型化目标 G
  -> build_belief(G)
       得到信念图对象
  -> belief.summarize_for_prompt(G)
       得到和目标相关的图事实
  -> retriever.topk(question)
       得到历史案例 R_t
  -> _build_messages(G, belief_summary, retrieved)
       拼接基础指令、少样本案例、类型化目标、信念快照
  -> engine.generate(..., n=n_hyp)
       采样多个候选程序
  -> verify(candidate, G, malt_graph)
       对每个候选生成证明对象
  -> 按通过状态和证明置信度排序
       选择最可信候选
  -> 如果失败：_build_repair_messages(...)
       把失败诊断信息注入修复提示词
  -> 最多重复修复 max_repair 轮
  -> 返回 VistaResult
```

### 2.2 批量 VISTA 运行流程

`src/run_vista.py` 用于单独跑 VISTA，分为两个阶段：

- 阶段 1 生成：加载测试集、训练集、检索器、大语言模型、MALT 图。对每个分块进行意图编译、信念构造、案例检索、多样本生成、验证和修复，把原始代码和证明写到 `outputs/raw_vista_base.jsonl` 或 `outputs/raw_vista_sft.jsonl`。
- 阶段 2 评分：调用 `score_records_parallel()`，用 NetArena 官方评估器评分，写出 `outputs/vista_base.jsonl` 或 `outputs/vista_sft.jsonl`，并生成 `.summary.json`。

`src/run_all_gen.py` 是更高效的统一生成运行器。它把多个基线和 VISTA 生成放在一个 vLLM 会话中，减少模型反复冷启动。

### 2.3 实验全流程

推荐实验脚本是 `scripts/run_full.sh`：

```text
src.data
  -> 生成训练/测试划分
src.run_all_gen
  -> 基础模型的基线方法 + vista_base 生成
src.score_all
  -> 并行评分已有原始输出
src.sft_train
  -> LoRA SFT 训练
src.run_all_gen --lora
  -> sft_only + vista_sft 生成
src.score_all
  -> 评分 LoRA 相关结果
src.aggregate
  -> 汇总 RESULTS.md 和 RESULTS.csv
```

---

## 3. 重点问题一：用户意图如何编译成结构化目标

意图编译的核心文件是 `src/intent.py`。

### 3.1 输入和输出

输入是 MALT 基准测试 的自然语言问题，例如：

```text
Add new node with name new_EK_PORT_82 type EK_PORT, to ju1.a2.m4.s3c6. Return a graph.
```

输出是一个 Python 字典，也就是类型化目标：

```python
{
    "task_type": "capacity_planning",
    "level": "level-1",
    "operations": [
        {
            "op": "add",
            "target": "new_EK_PORT_82",
            "params": {"type": "EK_PORT", "parent": "ju1.a2.m4.s3c6"}
        }
    ],
    "return_shape": "graph",
    "invariants": [
        "valid_node_types",
        "valid_edge_types",
        "valid_node_hierarchy",
        "no_isolated_nodes",
        "bandwidth_consistent",
        "ports_exist_under_switch"
    ],
    "allowed_actions": ["add"],
    "protected_scope": [],
    "stop_condition": {"predicate": "verified_pass and matches_return_shape"},
    "raw_query": "...",
    "parsed_steps": 1
}
```

这个 `Goal` 是后续所有模块共同使用的契约：

- 提示词 里会把 `Goal` 渲染成 `goal_to_hint()`。
- 验证器 会用 `Goal.return_shape` 检查输出类型。
- 验证器会用 `Goal.operations` 检查添加、删除、更新这些动作是否真的反映在更新后的图中。
- 修复提示词 会继续携带同一个 `Goal`，保证修复不会偏离原始意图。

### 3.2 正则模板如何覆盖 MALT 任务

`src/intent.py` 顶部定义了多个正则模板：

- `ADD_PAT`：匹配 一级添加。
- `REMOVE_PAT`：匹配删除操作。
- `COUNT_PAT`：匹配计数操作。
- `LIST_PAT`：匹配 列出子节点。
- `RANK_PAT`：匹配 对子节点排序。
- `UPDATE_PAT`：匹配更新物理容量。
- `ADD_THEN_PREFIX_PAT`：匹配复合任务中的添加前缀。
- `REMOVE_THEN_PREFIX_PAT`：匹配复合任务中的删除前缀。
- `COUNT_AFTER_PAT`、`LIST_AFTER_PAT`、`RANK_AFTER_PAT`：匹配 更新后的图 上的后续查询。

这些模板的设计是和 NetArena 的 `dy_query_generation.py` 任务模板对齐的。也就是说，VISTA 并不是让大语言模型自己抽取意图，而是利用基准测试可控语言的结构，做确定性解析。这是当前实现追求可复现和低噪声的选择。

### 3.3 `_parse_step(text)`

`_parse_step()` 是单句解析函数。它按顺序尝试各类正则：

- 如果匹配添加操作，返回 `{"op": "add", "target": name, "params": {"type": type, "parent": parent}}`。
- 如果匹配删除操作，返回 `{"op": "remove", "target": name, "params": {}}`。
- 如果匹配计数操作，返回 `{"op": "count", "target": None, "params": {"node_type": type, "scope": scope}}`。
- 如果匹配列表操作，返回 `{"op": "list", "target": None, "params": {"scope": scope}}`。
- 如果匹配排序操作，返回 `{"op": "rank", "target": None, "params": {"node_type": maybe_type, "scope": scope, "key": "physical_capacity_bps"}}`。
- 如果匹配更新操作，返回 `{"op": "update", "target": name, "params": {"attr": "physical_capacity_bps", "value": int(value)}}`。

函数返回的是一个“原子操作中间表示”。复合任务就是多个原子操作的顺序列表。

### 3.4 `_split_compound(query)`

复合任务通常由多个句子组成，例如：

```text
Add new_EK_PORT_3 to ju1.a1.m1.s2c1. Count the EK_PORT in ju1.a1 in the updated graph. Return the count number as text.
```

`_split_compound()` 用句号后接大写字母的位置拆分，保持顺序。顺序非常关键，因为 MALT 的 二级/三级 任务通常是先更新图，再在 更新后的图 上查询。

### 3.5 `compile_intent(query)`

`compile_intent()` 是意图编译的主函数，分三层解析：

1. 先尝试把整个查询当作一级单步任务解析。
2. 如果失败，再尝试复合前缀：添加后续 或 删除后续。
3. 如果还失败，就按句子拆分后逐句 `_parse_step()`。

之后它补齐几个关键字段：

- `return_shape`：如果出现 `Return a graph` 就是 `graph`；如果出现列表相关词就是 `list`；如果出现计数就是 `text`；否则默认 `graph`。
- `level`：`operations` 长度大于等于 3 或出现 `For each` 判为 `level-3`；长度等于 2 判为 `level-2`；否则 `level-1`。
- `invariants`：固定使用 MALT 安全检查中的六个不变量。
- `allowed_actions`：由解析出的操作类型集合得到。
- `stop_condition`：当前实现中固定为 `verified_pass and matches_return_shape`。

### 3.6 `goal_to_hint(goal)`

`goal_to_hint()` 把结构化目标渲染成注入提示词的注释文本，例如：

```text
# Compiled goal - task=capacity_planning, level=level-2, return_shape=text
#   step 1: op=add target=new_EK_PORT_3 - parent=ju1.a1.m1.s2c1
#   step 2: op=count - node_type=EK_PORT, scope=ju1.a1
# Invariants to preserve: valid_node_types, ...
```

这一步是 VISTA 的关键连接点：大语言模型 不只看到自然语言问题，还看到确定性解析后的操作序列和安全约束。它减少了模型自己误读任务的概率。

### 3.7 与方法论的关系

方法论文档中写：

```text
G = Compile(q, policy, role, environment)
G = {task_type, objective, invariants, risk_budget, allowed_actions, protected_scope, stop_condition}
```

当前实现中的 `Goal` 是这个公式的 MALT 简化版：

- `objective` 被具体化为 `operations` 和 `return_shape`。
- `risk_budget` 没有显式字段。
- `policy`、`role`、`environment` 没有外部输入，安全策略被固化为六个 MALT 不变量。
- `protected_scope` 在 MALT 中为空，因为 基准测试 没有明确 保护范围 概念。

---

## 4. 重点问题二：信念图如何构造，网络状态包含什么，不确定性如何量化

信念图的核心文件是 `src/belief.py`，底层真实网络图来自 `src/bridge.py::get_malt_graph()`。

### 4.1 方法论中的 信念图

方法论里，网络状态表示为：

```text
B_t = (V_t, E_t, X_t, U_t)
```

含义是：

- `V_t`：节点集合，包括设备、接口、链路、策略、路由、遥测窗口、故障事件、变更事件、工单、证据对象等。
- `E_t`：边集合，包括连接、策略应用、时间先后、证据支持、证据冲突等关系。
- `X_t`：节点和边上的类型化属性，比如容量、类型、时间戳、来源、访问域。
- `U_t`：不确定性元数据，比如 置信度、来源可靠性、新鲜度、冲突状态。

文档里的理想版本还包含 时序编码、图注意力聚合 和多时间尺度 摘要。

### 4.2 当前 MALT 实现中的 BeliefGraph

MALT 基准测试 已经提供了一个 `networkx.DiGraph`，所以当前实现没有重新构造完整图数据库，而是在真实图之上包一层 `BeliefGraph`：

```python
@dataclass
class BeliefGraph:
    G: nx.DiGraph
    evidence: dict[str, dict]
    assumptions: list[str]
    touched: set[str]
```

这四部分就是当前实现中的信念状态：

- `G`：真实 MALT 图，只读引用。它是完整拓扑状态。
- `evidence`：已经显式查询过的节点/边事实。当前主要记录目标节点或作用域节点是否存在、类型、孩子数量、孩子样例。
- `assumptions`：系统做出的假设，目前保留接口，实际较少使用。
- `touched`：被修改或计划修改的节点集合，目前保留接口，作为未来扩展点。

### 4.3 MALT 图中的网络状态包含哪些内容

底层 `G` 是 NetArena MALT 的 networkx 有向图，节点和边来自 `solid_step_helper.getGraphData()`。

在提示词和验证器中显式使用的状态包括：

- **节点名称**：例如 `ju1.a2.m4.s3c6`、`new_EK_PORT_82`。
- **节点类型**：如 `EK_JUPITER`、`EK_SUPERBLOCK`、`EK_AGG_BLOCK`、`EK_PACKET_SWITCH`、`EK_PORT`、`EK_CONTROL_DOMAIN`。
- **边类型**：主要是 `RK_CONTAINS` 和 `RK_CONTROLS`。
- **层级关系**：例如 `JUPITER -> SUPERBLOCK -> AGG_BLOCK -> PACKET_SWITCH -> PORT`。
- **容量属性**：尤其是 `physical_capacity_bps`，用于更新和排序。
- **孩子节点集合**：用于列表、排序、计数，也用于信念摘要。
- **安全不变量**：节点类型合法、边类型合法、层级合法、无孤立节点、带宽一致、交换机下有端口。

### 4.4 `BeliefGraph.has_node(name)`

该方法扫描 `G.nodes(data=True)`，判断节点编号或节点属性 `name` 是否等于输入名称。

用途：

- 支持验证器或提示词落地约束中的存在性判断。
- 避免依赖 networkx 内部节点编号，因为 MALT 图中节点编号和 `name` 属性都可能参与定位。

### 4.5 `BeliefGraph.find_node(name)`

该方法返回 `(node_id, attrs)`，其中 `attrs` 是 networkx 节点属性。

它是 `kind_of()`、`direct_children()` 和 `summarize_for_prompt()` 的基础函数。

### 4.6 `BeliefGraph.kind_of(name)`

该方法读取目标节点的 `type` 属性，并兼容两种格式：

- `type` 是列表，比如 `["EK_PORT"]`。
- `type` 是字符串，比如 `"EK_PORT"`。

返回值是单个类型字符串或 `None`。

### 4.7 `BeliefGraph.direct_children(name)`

该方法先找到目标节点，再遍历它的 出边，只保留 `type == RK_CONTAINS` 的边，返回直接子节点列表。

这一步是 信念快照 的核心，因为很多 MALT 任务都和直接孩子有关：

- `List direct child nodes ...`
- `Rank direct child nodes ...`
- `Count EK_PORT in scope ...`
- 添加或删除后检查局部层级是否合理。

### 4.8 `BeliefGraph.add_evidence(key, payload, conf=1.0)`

该方法把观察到的事实写入 `evidence`：

```python
self.evidence[key] = {
    "payload": payload,
    "confidence": conf
}
```

当前代码中，来自真实 图 的 证据 都用 `conf=1.0`。这是因为 MALT 是静态 基准测试，没有真实 遥测 噪声、采样延迟或多源冲突。

### 4.9 `BeliefGraph.assume(statement)`

该方法把假设写入 `assumptions`。当前主流程很少显式调用，但它对应方法论里的 未解决假设，未来可用于：

- 父节点是否存在但未确认。
- 某个容量属性是否可靠。
- 某个变更是否已经传播。
- 某个观察是否过期。

### 4.10 `BeliefGraph.summarize_for_prompt(goal)`

这是当前信念图最关键的落地函数。它接收类型化目标，针对每个操作中涉及的 `target` 和 `scope/parent` 查询真实图，并生成一段提示词注释：

```text
# Belief snapshot (queried evidence - fresh from the graph):
#   - node `ju1.a2.m4.s3c6`: exists, type=['EK_PACKET_SWITCH'], #children=...
#   - children-type distribution under `ju1.a2.m4`: {'EK_PACKET_SWITCH': 8}
```

函数做了几件事：

1. 从 `goal["operations"]` 中读取每个步骤。
2. 提取 `target`、`params.scope`、`params.parent`。
3. 对每个相关节点调用 `find_node()`。
4. 如果不存在，写入 `NOT FOUND` 证据，置信度 为 1.0。
5. 如果存在，记录类型、直接孩子数量、前 5 个孩子样例。
6. 对列表/排序操作额外统计作用域下直接孩子类型分布。
7. 把假设列表追加到摘要中。

这段 摘要 会通过 `planner._build_messages()` 拼到 系统提示词 中，让 大语言模型 在生成代码前看到当前网络状态。

### 4.11 `build_belief(goal)`

该函数从 `bridge.get_malt_graph()` 取真实图，构造新的 `BeliefGraph(G=G)`。

注意：当前实现每条 查询 都构造一个 新的信念封装，但底层 图 是 缓存单例。它不是跨 查询 持续演化的在线 智能体 状态，而是 基准测试 上每个样本独立评测的静态 快照。

### 4.12 不确定性如何量化

这里必须区分“方法论理想设计”和“当前代码实现”。

#### 方法论中的不确定性

完整 VISTA-Net++ 设计中，不确定性包含：

- 节点级 置信度。
- 证据 来源 和 新鲜度。
- 证据 支持/冲突。
- 假设后验 `p(h_k | B_t, G, R_t)`。
- 动作风险 `Risk(a | B_t)`。
- 信息增益 `IG(a; H_t | B_t)`。
- 反事实世界 `W_t = {(h_i, load_j, env_k)}`。

这些内容在文档中有公式化定义。

#### 当前代码中的不确定性

当前实现有三个实际量化位置：

1. **证据置信度**  
   `BeliefGraph.add_evidence(..., conf=1.0)`。由于 MALT 图是确定性 标准答案 图，观察到的节点事实 置信度 固定为 1.0。不存在的节点也被当成确定观察，置信度 也是 1.0。

2. **多假设不确定性**  
   `engine.generate(..., n=n_hyp, temperature=0.7)` 对同一 提示词 生成多个候选程序。这里没有显式计算 后验分布，但多个候选本身代表当前模型对可行程序的多个假设分支。

3. **证明置信度**  
   `verifier.verify()` 为每个候选计算加权 置信度：

   ```python
   weights = {
       "syntax_ok": 0.10,
       "type_ok": 0.10,
       "sim_executed": 0.20,
       "local_invariant_ok": 0.20,
       "semantic_eq_ok": 0.30,
       "sim_safe_ok": 0.10,
   }
   confidence = sum(weights[k] * float(getattr(p, k)) for k in weights)
   ```

   这个 置信度 是工程上的候选质量估计，体现“这个动作有多接近可安全执行”。它不是贝叶斯 后验，但承担了排序、选择、修复时的置信度作用。

#### 当前实现的诚实边界

当前代码没有显式实现：

- 多源遥测不确定性。
- 时间衰减 置信度。
- 证据冲突消解。
- 假设后验 softmax 归一化。
- 信息增益 公式。
- 多 负载/环境反事实世界。

但是它实现了 MALT 基准测试 上最重要的安全闭环：确定性 图 证据、多个候选分支、证明置信度、图副本模拟 和 不变量门控。

---

## 5. 重点问题三：如何根据历史案例和当前假设推理

历史案例和当前假设主要由 `src/retrieval.py`、`src/planner.py`、`src/run_vista.py` 和 `src/run_all_gen.py` 实现。

### 5.1 历史案例是什么

训练集中的每条记录都有：

```json
{
  "id": 164,
  "question": "Add new node ...",
  "answer": "def ground_truth_process_graph(graph_data): ...",
  "task_label": "capacity planning, level-1, add",
  "complexity": "level-1",
  "op": "add",
  "split": "train"
}
```

在 VISTA 中，这些记录不是普通知识段落，而是 过程型案例：

- `question` 是历史意图。
- `answer` 是可执行程序模板。
- `task_label`、`complexity`、`op` 是过程标签。
- 标准答案 函数会被改名为 `process_graph`，作为 少样本答案。

### 5.2 `_index_key_for(rec)`

`src/retrieval.py` 中的 `_index_key_for()` 返回：

```python
f"[{rec['complexity']}|{rec['op']}] {rec['question']}"
```

这很重要。检索不是只看问题文本，而是把复杂度和操作类型前缀放进嵌入向量文本，让检索更偏向过程相似：

- 二级删除后排序 更容易检索到 删除后排序。
- 添加后列表查询 更容易检索到 添加后列表查询。
- 排序任务更容易拿到排序辅助函数的示例。

这对应方法论中的 案例记忆检索：历史案例是 过程先验，不只是语义相似文本。

### 5.3 `CaseRetriever.build_or_load(records)`

该类方法负责构建或读取 FAISS 索引：

1. 根据 记录 的 `(id, task_label, question prefix)` 计算 缓存键。
2. 加载 `SentenceTransformer(C.EMBED_MODEL)`。
3. 如果缓存存在，读取 `.npy` 嵌入向量 和 `.bin` FAISS 索引。
4. 如果缓存不存在，对 `_index_key_for(r)` 生成的文本做 嵌入向量。
5. 使用 `faiss.IndexFlatIP` 做 余弦相似度 检索，因为 嵌入向量 已 归一化。
6. 保存 嵌入向量、索引 和 元数据。

### 5.4 `CaseRetriever.topk(query, k, tag=None)`

该方法把当前查询编码成嵌入向量，并在 FAISS 索引中搜索前 k 个相似项，返回训练集下标列表。

`tag` 是可选前缀，目前主流程没有强依赖。它可以用于未来把当前任务类型显式加入 查询嵌入向量。

### 5.5 `CaseRetriever.topk_records(query, k, tag=None)`

这是 `topk()` 的便捷封装，直接返回记录字典。

### 5.6 `_fewshot_from_records(records)`

`src/planner.py` 中该函数把检索到的历史记录转成提示词示例：

```python
{
    "question": r["question"],
    "answer": r["answer"].replace("ground_truth_process_graph", "process_graph").strip()
}
```

这一步把训练集答案从 基准测试 的 `ground_truth_process_graph` 标准函数名改成系统要求 大语言模型 输出的 `process_graph`。

### 5.7 `_build_messages(goal, belief_summary, retrieved, extra_hint=None)`

这个函数把三类信息合并到最终 提示词：

1. `P.build_messages(..., prompt_type="fewshot_semantic")` 构造基础提示词 + 语义少样本示例。
2. `goal_to_hint(goal)` 注入 类型化目标。
3. `belief_summary` 注入当前图状态 证据。
4. 如果是 修复，则追加 `extra_hint`。

最终 大语言模型 看到的是：

```text
系统角色和任务约束
+ 辅助函数说明
+ 图结构说明
+ 输出格式约束说明
+ 前 k 个历史案例
+ 已编译的类型化目标
+ 信念快照
+ 可选的修复诊断信息
```

### 5.8 当前假设如何表示

代码中的当前假设不是显式的 根因对象，而是候选程序集合：

```python
sampled = engine.generate(..., n=n_hyp, temperature=0.7)
```

每个样本 都是一个候选 `process_graph`，可以理解成：

- 对用户目标的一种解释。
- 对图操作顺序的一种实现。
- 对 辅助函数 调用方式的一种假设。
- 对返回格式的一种假设。

这些候选组成工程实现中的 `H_t`。

### 5.9 如何对假设进行推理和筛选

每个候选都会经过：

```python
code = P.extract_code_output(s["text"])
proof = verify(code, goal, malt_graph)
```

然后按：

```python
candidates.sort(key=lambda x: (x["proof"].passed, x["proof"].confidence), reverse=True)
```

排序。

这意味着：

- 通过 证明检查门 的候选优先。
- 都没通过时，置信度 高的候选优先，因为它可能只差一个局部问题。
- 失败候选会被用来构造 修复提示词。

### 5.10 证明引导修复 如何利用当前假设

如果所有候选都失败，VISTA 会选择当前最好的失败候选，把它的 证明诊断信息 注入 提示词：

````text
Your previous attempt failed verification.
- syntax_ok=...
- type_ok=...
- semantic_eq_ok=...
- local_invariant_ok=...
- ret_type=...
- updated_graph_present=...

Reason:
...

Previous attempt:
```python
...
```
````

这个过程相当于：

1. 先生成多个假设。
2. 用 验证器 找出最接近成功的假设。
3. 把失败原因变成下一轮条件。
4. 让 大语言模型 做局部修复而不是重新随机生成。

这就是当前代码里“根据当前假设推理”的主要落地。

---

## 6. 代码文件逐一讲解

本节按照文件路径逐个解释，每个函数和类都会说明职责、输入输出和在方法中的位置。

---

## 6.1 `methodology_v3_network_llm_clean.md`

这是方法论文档，不是可执行代码。它定义 VISTA-Net++ 的完整理论框架：

- 问题建模：将网络管理定义为部分可观测、安全约束下的序列决策问题。
- 核心视角：强调 持久信念状态，而不是 临时文本上下文。
- 框架总览：六阶段框架。
- 时序类型化信念图：定义 `B_t = (V_t, E_t, X_t, U_t)`。
- 信念编码器：定义多时间尺度时序编码器和图聚合。
- 案例记忆检索：定义历史案例 记录 和检索得分。
- 假设后验推理：定义 `p(h_k | B_t, G, R_t)`。
- 信息增益规划：定义动作效用，包括任务价值、信息增益、风险、成本、人工负担。
- 类型化动作构造：定义 `ActionIR`。
- 反事实评估和带证明的验证：定义 `W_t`、风险、通过门控和证明包。
- 训练目标：定义意图、检索、假设、规划、验证、等价性、校准、时序目标。
- 推理过程：定义循环推理过程。

这份文档是代码中注释引用 `§3`、`§4`、`§7-§12` 等章节的来源。

---

## 6.2 `src/config.py`

该文件集中定义路径、模型、超参数和目录创建逻辑。

### 常量说明

- `NETARENA_DIR`：NetArena 根目录。
- `NETARENA_MALT_DIR`：MALT 应用目录，用于导入 `solid_step_helper`、`malt_env`、`error_check`。
- `NETARENA_SRC_DIR`：NetArena 源代码目录。
- `VISTA_ROOT`：当前项目根目录。
- `DATA_DIR`、`MODELS_DIR`、`OUTPUTS_DIR`、`LOGS_DIR`、`CACHE_DIR`：数据、模型、输出、日志、缓存目录。
- `MALT_RAW_DATA`：原始 MALT 基准测试 JSONL 文件。
- `TRAIN_FILE`、`TEST_FILE`、`SFT_FILE`：训练集、测试集和 SFT 数据路径。
- `BASE_MODEL`：Qwen2.5-Coder-7B-Instruct 本地路径。
- `EMBED_MODEL`：BGE 嵌入向量 模型路径。
- `SFT_OUTPUT_DIR`：LoRA SFT 输出路径。
- `TEST_PER_LABEL`：每个任务标签取 100 条测试样本。
- `TRAIN_PER_LABEL`：每个任务标签取 400 条训练样本。
- `SEED`：数据切分随机种子。
- `VLLM_GPU_MEMORY_UTILIZATION`、`VLLM_MAX_MODEL_LEN`、`VLLM_DTYPE`：vLLM 加载配置。
- `GEN_MAX_TOKENS`、`GEN_TEMPERATURE_GREEDY`、`GEN_TEMPERATURE_SAMPLE`、`GEN_TOP_P`：生成配置。
- `VISTA_NUM_HYPOTHESES`：每条查询采样候选数量，默认 5。
- `VISTA_MAX_REPAIR_ITER`：最大修复轮数，默认 3。
- `VISTA_RETRIEVAL_K`：历史案例检索数量，默认 4。
- `EVAL_LIMIT`：评测样本限制，默认全量。

### 模块尾部目录创建

文件最后对 `DATA_DIR`、`MODELS_DIR`、`OUTPUTS_DIR`、`LOGS_DIR`、`CACHE_DIR` 执行 `os.makedirs(..., exist_ok=True)`。这保证导入配置时必要目录存在。

---

## 6.3 `src/data.py`

该文件负责数据准备：读取 NetArena 原始 MALT 数据，做分层切分，写出 VISTA 使用的训练/测试 JSONL。

### `Example`

`Example` 是数据样本数据类：

- `id`：原始行号。
- `question`：自然语言任务。
- `answer`：标准答案 Python 代码，函数名是 `ground_truth_process_graph`。
- `task_label`：任务标签，例如 `capacity planning, level-1, add`。
- `complexity`：复杂度等级，如 `level-1`。
- `op`：操作类型，如 `add`、`remove-list`。
- `split`：`train` 或 `test`。

#### `Example.to_json()`

返回 `asdict(self)`，用于写 JSONL。

### `_parse_label(task_label)`

把 `task_label` 按逗号拆分：

- 正常情况下返回 `(parts[1], parts[2])`，即复杂度和操作类型。
- 如果格式不完整，做 兜底，避免崩溃。

### `load_raw(path=C.MALT_RAW_DATA)`

读取原始 NetArena JSONL。每行中 `messages` 包含 `question`、`answer`、`task_label`。函数做的事情：

1. 逐行 `json.loads`。
2. 遍历 `messages` 找出 `question`、`answer`、`task_label`。
3. 用 `_parse_label()` 得到复杂度和操作类型。
4. 构造 `Example`。

返回 `list[Example]`。

### `stratified_split(examples, test_per_label, train_per_label, seed)`

按 `task_label` 分层切分数据：

1. 用 `defaultdict(list)` 按 标签 分桶。
2. 用固定随机种子打乱每个桶。
3. 每个标签最后 `test_per_label` 条作为测试集。
4. 前面最多 `train_per_label` 条作为训练集。
5. 给每个样本标记 `split`。

这样保证 10 个任务范围在测试集中每类数量一致，评测更公平。

### `write_jsonl(records, path)`

把 `Example` 列表逐行写成 JSON。

### `read_jsonl(path)`

读取 JSONL，返回 `list[dict]`。项目里几乎所有运行器都用它读训练集、测试集和原始输出。

### `main()`

命令行入口：

1. `load_raw()`。
2. 打印样本数和 标签 分布。
3. `stratified_split()`。
4. 写 `C.TRAIN_FILE` 和 `C.TEST_FILE`。

运行方式：

```bash
python -m src.data
```

---

## 6.4 `src/prompts.py`

该文件定义 提示词 模板和 大语言模型 输出代码抽取逻辑。

### `BASE_PROMPT`

基础 系统提示词。它告诉模型：

- 作为网络工程师处理容量规划图。
- 输出 `def process_graph(graph_data)`。
- 优先使用 NetArena 辅助函数：
  - `solid_step_add_node_to_graph`
  - `solid_step_remove_node_from_graph`
  - `solid_step_counting_query`
  - `solid_step_list_child_nodes`
  - `solid_step_update_node_value`
  - `solid_step_rank_child_nodes`
- 图 是 `networkx.DiGraph`。
- 节点类型和边类型有哪些。
- 更新图时要深拷贝。
- 返回对象必须包含 `type`、`data`、`updated_graph`。
- 只能输出一个带围栏的 Python 代码块。

这是所有 基线 和 VISTA 都共用的底层任务规范。

### `COT_HINT`

用于 `zeroshot_cot` 和 `fewshot_cot`，要求模型先思考操作类型、目标节点和安全问题，再输出最终代码块。

注意：最终提取仍只提取 Python 代码块。

### `EXAMPLE_LIST`

固定少样本示例列表，覆盖更新、添加、计数、删除、列表、排序。每个示例答案都使用规范的 `solid_step_*` 辅助函数，降低模型生成偏离基准测试语义的概率。

### `USER_QUERY_TEMPLATE`

把具体用户问题包成用户消息，强调：

- 只输出一个 Python 代码块。
- 必须定义 `process_graph(graph_data)`。
- 返回字典必须有 `type`、`data`、`updated_graph`。

### `_format_examples(exs)`

把 少样本示例 格式化成 提示词 文本：

````text
Question: ...

Answer:
```python
...
```
````

### `build_messages(question, prompt_type, fewshot_examples=None)`

生成 ChatML 消息。支持：

- `zeroshot_base`
- `zeroshot_cot`
- `fewshot_base`
- `fewshot_cot`
- `fewshot_semantic`

处理逻辑：

1. `sys_parts` 从 `BASE_PROMPT` 开始。
2. 如果是思维链提示词，追加 `COT_HINT`。
3. 如果是固定少样本，追加 `EXAMPLE_LIST`。
4. 如果是 语义少样本，必须传入 `fewshot_examples`。
5. 拼出系统消息和用户消息。

VISTA 的 `planner._build_messages()` 会在这个基础上继续追加 类型化目标 和 信念快照。

### `CODE_FENCE_RE`

正则，用于提取 围栏代码块：

````python
```(?:python)?\s*(.*?)```
````

### `PROCESS_GRAPH_DEF_RE`

正则，用于把各种类似 `xxxprocess_graphxxx` 的函数名归一化为 `process_graph`。

### `extract_code_output(answer)`

从大语言模型输出中鲁棒抽取代码：

1. 优先找 围栏代码块。
2. 选择包含 `process_graph` 的最长 代码块。
3. 如果没有围栏代码块，就从文本中 `def process_graph` 位置开始截取。
4. 统一函数名为 `process_graph`。
5. 去掉 SFT 可能输出的 HTML 注释，例如 `<!-- intent: ... -->`。
6. 删除 `import ...` 行，因为 NetArena 规则要求不能导入额外模块。

该函数是 生成 和 评分 中间的关键清洗层。

---

## 6.5 `src/intent.py`

该文件已经在第 3 节详细解释。这里按函数清单总结。

### 常量和正则

- `ID`：节点名 token 模式。
- `QID`：可选引号的节点名模式。
- `ADD_PAT`、`REMOVE_PAT`、`COUNT_PAT`、`LIST_PAT`、`RANK_PAT`、`UPDATE_PAT`：单步任务模板。
- `ADD_THEN_PREFIX_PAT`、`REMOVE_THEN_PREFIX_PAT`：复合任务前缀。
- `COUNT_AFTER_PAT`、`LIST_AFTER_PAT`、`RANK_AFTER_PAT`：更新后的图查询模板。

### `_parse_step(text)`

把一句自然语言解析为一个操作字典。

### `_split_compound(query)`

按句子边界切分复合任务，保持操作顺序。

### `compile_intent(query)`

主编译函数，输出 类型化目标。

### `goal_to_hint(goal)`

把 类型化目标 渲染为 提示词 注释。

### `_smoke()`

本地冒烟测试，打印几个样例的已编译目标。

---

## 6.6 `src/bridge.py`

该文件是 VISTA 和 NetArena 官方 MALT 代码之间的桥。

### 模块级导入和路径注入

文件开头把：

- `C.NETARENA_MALT_DIR`
- `C.NETARENA_SRC_DIR`

插入 `sys.path`，然后导入 NetArena 的：

- `getGraphData`
- `validate_llm_output`
- `clean_up_updated_graph_data`
- `solid_step_*` 辅助函数
- `SafetyChecker`
- `BenchmarkEvaluator`

这保证 VISTA 的执行和评分与 NetArena 官方实现一致。

### `_GRAPH_TUPLE`

模块级 单例 缓存。避免每次调用都重新解析 MALT graph。

### `get_malt_graph()`

如果 `_GRAPH_TUPLE` 为空，调用 `getGraphData()` 加载真实 MALT 图；之后返回 `_GRAPH_TUPLE[1]`，也就是 `nx.DiGraph`。

### `make_evaluator()`

用 cached graph 构造 `BenchmarkEvaluator(graph_data=get_malt_graph())`。

### `score_one(query, golden_answer, llm_code, task_label, run_latency=None, evaluator=None)`

用 NetArena 官方评估器评分单条样本：

1. 如果没有传入评估器，调用 `make_evaluator()`。
2. 调用 `ev.run_agent_output(query, golden_answer, llm_answer=llm_code)`。
3. 捕获异常，异常时 正确性 和 安全性 都设为 Fail。
4. 调用 `ev.ground_truth_check(...)` 得到官方结果。

返回字典，包含：

- `Result-Correctness`
- `Result-Safety`
- `Result-Latency`
- `Label`
- `Query`
- optional `Error`

### `_smoke()`

读取第一条训练样本，把 标准答案 code 改名为 `process_graph` 后评分。理论上 正确性 应该 Pass，用于检查桥接是否正常。

---

## 6.7 `src/belief.py`

该文件已经在第 4 节详细解释。这里按类和方法总结。

### `BeliefGraph`

字段：

- `G`：真实 MALT `nx.DiGraph`。
- `evidence`：目标相关证据表。
- `assumptions`：假设列表。
- `touched`：修改或计划修改的节点集合。

### `BeliefGraph.has_node(name)`

判断图中是否存在指定 name。

### `BeliefGraph.find_node(name)`

返回节点 id 和属性。

### `BeliefGraph.kind_of(name)`

返回节点类型字符串。

### `BeliefGraph.direct_children(name)`

返回 `RK_CONTAINS` 直接孩子节点。

### `BeliefGraph.add_evidence(key, payload, conf=1.0)`

记录 证据 和 置信度。

### `BeliefGraph.assume(statement)`

记录假设。

### `BeliefGraph.summarize_for_prompt(goal)`

根据 类型化目标 查询相关节点，把目标相关图状态写成 提示词片段。

### `build_belief(goal)`

加载真实 MALT 图，返回新的 `BeliefGraph` wrapper。

---

## 6.8 `src/retrieval.py`

该文件实现 案例记忆检索。

### `_index_key_for(rec)`

将记录格式化为嵌入向量文本：

```python
[{complexity}|{op}] {question}
```

这样做使检索偏向相似过程，而不只是相似词面。

### `CaseRetriever`

字段：

- `records`：训练集记录。
- `embedder`：SentenceTransformer。
- `index`：FAISS 索引。
- `embeddings`：训练集 嵌入向量。

### `CaseRetriever.__init__(records, embedder, index, embeddings)`

保存 记录、嵌入向量 模型、FAISS 索引 和 嵌入向量 matrix。

### `CaseRetriever.build_or_load(records, cache_dir=None)`

构建或读取 嵌入向量 缓存 和 FAISS 索引：

- 使用 `C.EMBED_MODEL`。
- 使用 md5 hash 管理缓存失效。
- 嵌入向量 时 `normalize_embeddings=True`。
- 索引 使用 `faiss.IndexFlatIP`。

### `CaseRetriever.topk(query, k=C.VISTA_RETRIEVAL_K, tag=None)`

把当前查询编码，搜索前 k 个相似训练样本，返回索引列表。

### `CaseRetriever.topk_records(query, k=C.VISTA_RETRIEVAL_K, tag=None)`

返回前 k 个记录字典。

---

## 6.9 `src/llm_engine.py`

该文件封装 vLLM，所有基线、VISTA 生成和修复都通过这里调用模型。

### `LLMEngine`

负责加载 Qwen2.5-Coder-7B-Instruct，支持可选 LoRA。

### `LLMEngine.__init__(...)`

主要参数：

- `model_path`
- `gpu_memory_utilization`
- `max_model_len`
- `dtype`
- `enable_lora`
- `max_loras`
- `max_lora_rank`
- `seed`
- `download_dir`

执行逻辑：

1. 延迟导入 `vllm.LLM` 和 `SamplingParams`。
2. 设置 `TOKENIZERS_PARALLELISM=false`。
3. 创建 vLLM `LLM(...)`。
4. 保存分词器和 `SamplingParams` 类。

### `LLMEngine._make_sampling(**kw)`

创建 vLLM `SamplingParams`：

- 默认贪心生成：temperature 0。
- 默认 top_p、max_tokens 来自 config。
- 外部传入参数覆盖默认值。

### `LLMEngine.chat_template(messages)`

调用分词器的 `apply_chat_template()`，把 ChatML 消息转成模型提示词。

### `LLMEngine.generate(messages_batch, ...)`

项目统一的生成入口。

输入：

- `messages_batch`：一批消息列表。
- `temperature`、`top_p`、`max_tokens`、`n`。
- `lora_request`：可选 LoRA。

输出：

```python
[
    [
        {"text": sample_text, "latency_s": per_query_lat},
        ...
    ],
    ...
]
```

注意：vLLM 是批量并行生成，所以函数把总耗时均分给每个查询，作为近似延迟。

### `make_lora_request(adapter_path, lora_id=1, lora_name="sft")`

构造 vLLM `LoRARequest`，供 `LLMEngine.generate(..., lora_request=...)` 使用。

---

## 6.10 `src/planner.py`

该文件实现单 查询 版本的 VISTA planner。

### `VistaResult`

数据类，保存一次 VISTA 推理结果：

- `final_code`
- `final_response`
- `proof`
- `proof_passed`
- `n_hypotheses_tried`
- `n_repair_iterations`
- `goal`
- `belief_summary`
- `repair_history`
- `total_latency_s`

### `_fewshot_from_records(records)`

把检索记录转成少样本示例，并把函数名改成 `process_graph`。

### `_build_messages(goal, belief_summary, retrieved, extra_hint=None)`

构造 VISTA 提示词：

1. 用 语义少样本 构造基础 消息。
2. 追加 `goal_to_hint(goal)`。
3. 追加 `belief_summary`。
4. 追加可选修复提示。

### `_build_repair_messages(goal, belief_summary, retrieved, prev_code, prev_proof)`

构造修复提示词。它把失败候选的证明诊断信息、修复提示和上一次尝试都写进去。

### `run_vista_one(...)`

单条 查询 的完整 VISTA 流程：

1. `compile_intent(question)`。
2. `build_belief(goal)`。
3. `belief.summarize_for_prompt(goal)`。
4. `retriever.topk(question)`。
5. `_build_messages(...)`。
6. `engine.generate(..., n=n_hyp)` 采样多个候选。
7. 对每个候选 `extract_code_output()` 和 `verify()`。
8. 按 `proof.passed` 和 `proof.confidence` 排序。
9. 如果通过，返回。
10. 如果失败，进入最多 `max_repair` 轮修复。
11. 修复中如果新候选通过或置信度更高，替换当前最优候选。
12. 返回最终 `VistaResult`。

该函数是理解 VISTA 推理逻辑最清晰的入口。

---

## 6.11 `src/verifier.py`

该文件实现带证明的验证，是安全闭环的核心。

### `Proof`

数据类，记录候选程序的验证结果。

布尔检查字段：

- `syntax_ok`
- `type_ok`
- `local_invariant_ok`
- `global_invariant_ok`
- `semantic_eq_ok`
- `sim_safe_ok`
- `sim_executed`

诊断字段：

- `syntax_error`
- `type_error`
- `invariant_error`
- `semantic_error`
- `sim_error`

模拟输出摘要：

- `sim_ret_type`
- `sim_ret_data_kind`
- `sim_ret_data_summary`
- `sim_updated_graph_present`

最终字段：

- `passed`
- `confidence`
- `repair_hint`
- `elapsed_s`

#### `Proof.to_json()`

返回 `asdict(self)`，便于写 JSONL。

### `_check_syntax(code)`

用 `ast.parse(code)` 检查 Python 语法。如果空代码或 SyntaxError，返回失败原因。

### `_check_signature(code)`

确认代码里存在 `def process_graph(...)`，且至少有一个 positional argument。

### `_summary_of_data(data, max_len=120)`

把模拟返回的 `data` 简短序列化，写入证明诊断信息。

### `_safety_check(graph)`

调用 NetArena 的 `SafetyChecker(ret_graph=graph).evaluate_all()`，检查六个图不变量。

### `_semantic_check(ret, goal)`

检查候选程序返回结果是否符合 类型化目标：

1. `ret` 必须是字典。
2. 必须有 `type` 和 `data`。
3. `ret["type"]` 必须和 `goal["return_shape"]` 匹配。
4. 如果有 `updated_graph`，尝试转回 networkx 图。
5. 对 `goal.operations` 逐步检查：
   - 添加：目标节点必须出现在更新后的图中。
   - 删除：目标节点不能再出现在更新后的图中。
   - 更新：目标节点属性值必须等于目标值。

对计数/列表/排序的语义，目前更多依赖 NetArena 最终正确性评分器；验证器主要确保返回形态和图修改语义。

### `_exec_namespace_for(graph_data)`

构造执行候选代码时的 命名空间：

- `G`
- `copy`
- `nx`
- `json`
- 所有 `solid_step_*` 辅助函数

这样大语言模型生成的代码即使没有导入语句，也能直接调用辅助函数。

### `verify(code, goal, malt_graph)`

完整验证流程：

1. 创建 `Proof`。
2. 语法检查。
3. 函数签名检查。
4. 在 `copy.deepcopy(malt_graph)` 上执行候选代码：
   - `exec(code, ns)`
   - `eval("process_graph(copy.deepcopy(G))", ns)`
5. 检查返回对象是否为字典。
6. 提取返回类型、数据类型、数据摘要、是否有更新后的图。
7. 对更新后的图做安全性检查。
8. 做语义检查。
9. 计算 `sim_safe_ok`。
10. 计算总 `passed`。
11. 计算加权置信度。
12. 如果失败，生成 `repair_hint`。

这是 VISTA 中“每个执行前反事实模拟和验证”的实际代码实现。由于执行发生在图副本上，不会污染真实基准测试图。

---

## 6.12 `src/polish.py`

该文件实现类型化输出归一化。它在 `run_all_gen.py` 的 VISTA 生成路径中使用。

### `_detect_graph_var(fn)`

在 AST 中寻找常见图变量名：

- `graph_copy`
- `graph_data`
- `G`
- `graph`

找不到则 兜底 到函数第一个参数名。

### `polish_code(code)`

如果代码缺少 `updated_graph`，尝试在最终 `return` 前注入 格式约束 补齐逻辑：

- 序列化图变量为 node-link 数据。
- `return_object.setdefault('updated_graph', __vista_jg)`。
- 补齐 `type`。
- 补齐 `data`。

如果代码不可解析、没有 `process_graph`、已经有 `updated_graph`、找不到 return，就保持原样。

这对应方法论中的类型化动作构造：在进入验证器前把输出结构尽量标准化。

### `_smoke()`

打印规整前后代码，检查本来就合格的代码是否不变。

---

## 6.13 `src/parallel_score.py`

该文件并行调用 NetArena 评分器。它主要解决官方评估器在大图同构检查上很慢的问题。

### `_EVALUATOR`

工作进程内的全局评估器缓存。

### `_worker_init()`

每个 工作进程 启动时：

1. 插入 NetArena 路径。
2. 导入 `BenchmarkEvaluator` 和 `getGraphData`。
3. 加载 MALT graph。
4. 创建 `_EVALUATOR`。

### `_score_one_in_worker(args)`

评分单条 记录：

1. 读取 `query`、`golden`、`code`、`label`、`latency`。
2. 调用 `_EVALUATOR.run_agent_output(...)`。
3. 调用 `_EVALUATOR.ground_truth_check(...)`。
4. 捕获异常并返回 Fail。
5. 附加内部字段：
   - `__id`
   - `__extracted_code`
   - `__raw_response`

### `score_records_parallel(records, num_workers=8, timeout_per_task=120.0)`

主并行评分函数：

1. 使用 multiprocessing `spawn` context。
2. 创建 `ProcessPoolExecutor`。
3. 每条 记录 submit 到 工作进程。
4. 用 `as_completed()` 收集结果。
5. 保持输出顺序与输入顺序一致。
6. 工作进程 异常时返回 Fail 记录。

### `main()`

CLI 用法：

```bash
python -m src.parallel_score in.jsonl out.jsonl [num_workers]
```

读取 原始 JSONL，转换为 评分器 记录，调用 `score_records_parallel()`，写输出 JSONL。

---

## 6.14 `src/run_baseline.py`

该文件运行 基线 提示词 策略。

### `_build_messages_for_baseline(name, question, retriever=None, train_pool=None)`

根据基线名称构造提示词：

- `zeroshot_base`、`zeroshot_cot`、`fewshot_base`、`fewshot_cot`：直接调用 `prompts.build_messages()`。
- `fewshot_semantic`：先用检索器找前 k 个相似案例，再构造语义少样本。
- `sft_only`：使用基础提示词，但生成时传 LoRA。

### `_gt_to_process_graph(gt_code)`

把 标准答案 函数名替换为 `process_graph`，用于 少样本示例。

### `_load_train_pool()`

读取 `C.TRAIN_FILE`。

### `phase1_generate(name, test, raw_out_path, batch_size=16, lora_path=None)`

基线 的 GPU 生成阶段：

1. 如果是语义少样本，构建检索器。
2. 加载 `LLMEngine`，可选 LoRA。
3. 分批构造提示词。
4. 贪心生成，`n=1`。
5. `extract_code_output()` 抽取代码。
6. 写 `raw_<name>.jsonl`。

### `phase2_score(raw_path, out_path, num_workers=12)`

基线 的 CPU 评分 阶段：

1. 读取 原始 JSONL。
2. 调用 `score_records_parallel()`。
3. 写已评分 JSONL。
4. 统计 正确性、安全性、延迟。
5. 写 `.summary.json`。

### `run_baseline(...)`

组合第一阶段和第二阶段的便捷函数。

### `main()`

CLI 入口，支持：

- `--name`
- `--test`
- `--out`
- `--raw-out`
- `--limit`
- `--batch-size`
- `--lora`
- `--num-workers`
- `--phase all|gen|score`

---

## 6.15 `src/run_vista.py`

该文件是单独运行 VISTA 的批量运行器。

### `_process_one_initial(question, retriever, train_pool, malt_graph)`

对一条问题做 VISTA 初始上下文准备：

1. `compile_intent(question)`。
2. `build_belief(goal)`。
3. `belief.summarize_for_prompt(goal)`。
4. `retriever.topk(question)`。
5. `_build_messages(goal, belief_summary, retrieved)`。

返回 `(goal, belief_summary, retrieved, msgs)`。

### `phase1_vista_generate(...)`

VISTA 生成 主函数：

1. 读取测试集和训练样本池。
2. 构建 `CaseRetriever`。
3. 加载 `LLMEngine`，可选 LoRA。
4. 加载 MALT 图。
5. 按分块处理测试集。
6. 对每条查询做意图编译、信念构造、案例检索和消息构造。
7. 批量 `engine.generate(..., n=n_hyp, temperature=0.7)`。
8. 对每个样本抽取代码并 `verify()`。
9. 选择当前最优候选。
10. 对失败查询做最多 `max_repair` 轮贪心修复。
11. 写原始输出，包括代码、证明、目标和信念摘要。

### `phase2_vista_score(raw_path, out_path, num_workers, method)`

读取 VISTA 原始输出，用 `score_records_parallel()` 做官方评分，并把证明、目标和信念摘要一起带到已评分输出。

### `main()`

CLI 入口，支持：

- `--test`
- `--out`
- `--raw-out`
- `--limit`
- `--chunk-size`
- `--n-hyp`
- `--max-repair`
- `--lora`
- `--num-workers`
- `--phase all|gen|score`

---

## 6.16 `src/run_all_gen.py`

该文件是统一 生成 运行器。它的目标是只加载一次 vLLM，然后连续跑多个 生成阶段。

### `_gt_to_process_graph(gt_code)`

和 基线 中一样，把 标准答案 函数名改成 `process_graph`。

### `_build_messages_for_baseline(name, question, retriever=None, train_pool=None)`

统一运行器版本的基线提示词构造器。

### `_gen_baseline(engine, name, test, lora_req, batch_size, raw_path, retriever, train_pool)`

使用已有 `engine` 生成 基线 原始输出，避免重复加载模型。

### `_gen_vista(engine, method, test, lora_req, chunk_size, n_hyp, max_repair, raw_path, retriever, train_pool, malt_graph)`

统一运行器版本的 VISTA 生成。和 `run_vista.py` 相比，多了：

- 对 原始代码 先调用 `polish_code(raw_code)`。
- 修复后也调用 `polish_code()`。

因此 `run_all_gen.py` 是当前更完整的 VISTA 生成 实现。

### `main()`

执行逻辑：

1. 读取测试集和训练样本池。
2. 构建检索器和 MALT 图。
3. 加载 vLLM 引擎，可选 LoRA。
4. 决定要跑哪些基线。
5. 跑基础提示词基线。
6. 如果有 LoRA，跑 `sft_only`。
7. 跑 `vista_base`。
8. 如果有 LoRA，跑 `vista_sft`。

支持参数：

- `--limit`
- `--batch-size`
- `--vista-chunk-size`
- `--n-hyp`
- `--max-repair`
- `--lora`
- `--baselines`
- `--skip-vista`

---

## 6.17 `src/score_all.py`

该文件批量评分 `outputs/raw_*.jsonl`。

### `_is_baseline_prompt(name)`

判断名称是否属于基线：

- `zeroshot_base`
- `zeroshot_cot`
- `fewshot_base`
- `fewshot_cot`
- `fewshot_semantic`
- `sft_only`

### `_out_path_for(name)`

决定已评分输出路径：

- 基线 写 `outputs/baseline_<name>.jsonl`。
- VISTA 写 `outputs/<name>.jsonl`。

### `_score_one(raw_path, num_workers)`

评分一个 原始 JSONL：

1. 从文件名解析方法名。
2. 如果输出和摘要已存在且比原始文件新，就跳过。
3. 读取 记录。
4. 调用 `score_records_parallel()`。
5. 写已评分 JSONL。
6. 保留原始输出中的 `proof`、`goal`、`belief_summary`。
7. 写摘要 JSON。

### `main()`

扫描 `outputs/raw_*.jsonl`，可用 `--names` 限制评分对象。逐个调用 `_score_one()`。

---

## 6.18 `src/aggregate.py`

该文件汇总所有 `.summary.json` 和已评分 JSONL，生成结果表。

### `METHOD_ORDER`

定义结果展示顺序。

### `_load_results_jsonl(p)`

读取已评分 JSONL。

### `aggregate(out_dir=C.OUTPUTS_DIR)`

汇总流程：

1. 扫描 `*.summary.json`。
2. 规范化 方法名。
3. 读取对应 `.jsonl`。
4. 统计整体指标。
5. 按 `Label` 统计逐任务指标。
6. 写 `RESULTS.md`。
7. 写 `RESULTS.csv`。

### `main()`

CLI 入口，支持 `--out-dir`。

---

## 6.19 `src/sft_train.py`

该文件训练 Qwen2.5-Coder-7B-Instruct 的 LoRA 适配器。

### `_build_chat_dataset(records, tokenizer)`

把训练记录转成 SFT 文本：

1. 将标准答案代码的函数名改为 `process_graph`。
2. 在助手代码块中加入意图头：

   ```html
   <!-- intent: task_type=level-1; op=add -->
   ```

3. 用 `P.build_messages(question, prompt_type="zeroshot_base")` 构造系统消息和用户消息。
4. 追加助手消息。
5. 用分词器聊天模板渲染完整训练文本。

这里的意图头是一种轻量意图监督。它让模型在微调时显式看到任务等级和操作类型。

### `main()`

训练流程：

1. 解析训练参数。
2. 加载分词器和基础模型。
3. 读取 训练集划分。
4. 构造 数据集。
5. 开启梯度检查点。
6. 配置 LoRA：
   - `q_proj`
   - `k_proj`
   - `v_proj`
   - `o_proj`
   - `gate_proj`
   - `up_proj`
   - `down_proj`
7. 配置 `SFTConfig`。
8. 创建 `SFTTrainer` 并训练。
9. 保存最终适配器和分词器。

这对应方法论里的 `L_intent + L_plan` 的工程实现，其中 `L_plan` 是对标准答案代码的交叉熵训练，`L_intent` 通过意图头间接加入。

---

## 6.20 `src/reextract.py`

该文件用于在 `extract_code_output()` 逻辑变化后，重新提取 原始 JSONL 中的 `code` 字段。

### `main()`

1. 从命令行读取一个或多个 原始 JSONL 路径。
2. 逐条读取 `raw_response`。
3. 用当前 `extract_code_output()` 重新抽取代码。
4. 统计变化数量。
5. 覆盖写回原文件。

典型用途：当抽取器增加了 HTML 注释剥离之后，不用重新生成大语言模型输出，只需重抽代码。

---

## 6.21 `src/__init__.py`

包根文件。模块说明表达两层含义：VISTA-Net++ 是可验证、由信息增益引导、带类型增强的网络智能体；当前代码是在 NetArena 的 MALT 基准测试之上做的务实工程实现。

它让 `src` 成为 Python 包，支持 `python -m src.xxx`。

---

## 6.22 `_smoke_prompts.py`

这是 提示词 冒烟测试 脚本。

主要逻辑：

1. 导入 `src.prompts`。
2. 构造一个 `zeroshot_cot` 提示词。
3. 打印系统/用户消息内容片段。
4. 构造一个含 围栏代码块 的模拟回答。
5. 测试 `prompts.extract_code_output()`。
6. 构造 `fewshot_base` 提示词，打印系统消息长度。

用途是快速检查提示词构造器和代码抽取器是否正常。

---

## 6.23 `run_vista_base_then_score.sh`

这是一个辅助 shell 脚本，用于等待 `vista_sft` 生成结束后跑 `vista_base` 和评分。

流程：

1. `cd /home/ubuntu/vista`。
2. `pgrep -f "run_all_gen.*vista_sft"` 循环等待。
3. 跑：

   ```bash
   python -m src.run_all_gen --baselines vista_base --vista-chunk-size 8 --n-hyp 5 --max-repair 3
   ```

4. 跑：

   ```bash
   python -m src.score_all --num-workers 12 --names vista_sft vista_base
   ```

5. 跑 `python -m src.aggregate`。

这是实验恢复或补跑场景下的脚本。

---

## 6.24 `scripts/run_all.sh`

早期端到端编排脚本。

流程：

1. `src.data` 生成数据。
2. 依次跑提示词基线：
   - `zeroshot_base`
   - `zeroshot_cot`
   - `fewshot_base`
   - `fewshot_semantic`
3. 跑 `src.sft_train`。
4. 用 LoRA 跑 `sft_only`。
5. 跑 `vista_base`。
6. 跑 `vista_sft`。
7. 跑 `src.aggregate`。

它每一步都可能单独加载 vLLM，整体效率不如 `run_full.sh`。

---

## 6.25 `scripts/run_pipeline.sh`

顺序流水线脚本，通过 `tee` 把日志写到 `logs/run_pipeline.log`。

流程类似 `run_all.sh`，但更强调：

- 每个阶段打印分隔线。
- 训练后检查 SFT 适配器 是否存在。
- 如果 最终适配器 不存在，尝试找 checkpoint。
- 如果没有 LoRA，就跳过 LoRA-dependent runs。
- 最后 aggregate。

---

## 6.26 `scripts/run_full.sh`

当前更推荐的 完整流水线。

特点：

- 使用 `src.run_all_gen`，单个 vLLM session 内连续跑多个 生成，减少冷启动。
- 生成 和 评分 分离。
- 中间先 评分 阶段-A 输出，即使后续 SFT 失败，也能保留已有结果。
- LoRA 相关 生成 放在第二个 vLLM session 中。
- 最终统一 aggregate。

流程：

1. `src.data`。
2. `src.run_all_gen` 跑基础模型基线和 `vista_base`。
3. `src.score_all`。
4. `src.sft_train`。
5. 找 SFT 适配器。
6. `src.run_all_gen --lora ... --baselines sft_only vista_sft`。
7. `src.score_all`。
8. `src.aggregate`。

---

## 6.27 `scripts/resume_from_sft.sh`

从 SFT 阶段恢复的脚本。

流程：

1. 跑 `src.sft_train`。
2. 找 SFT 适配器 或 checkpoint。
3. 跑 `src.run_all_gen --lora ... --baselines sft_only vista_sft`。
4. 跑 `src.score_all`。
5. 跑 `src.aggregate`。

适合 base 生成 已经完成，只需要重新做 LoRA 后半段的情况。

---

## 7. 证明对象和安全验证如何对应我们的方案

VISTA 的方案不是“生成完就评分”，而是“候选动作执行前先自证安全”。代码中体现为两级验证：

### 7.1 内部验证器

`src/verifier.py::verify()` 在 VISTA 生成阶段运行。它不做昂贵的图同构检查，而是快速检查：

- Python 语法。
- 函数签名。
- 在 图副本 上能否执行。
- 返回字典格式约束。
- 更新后的图是否满足安全不变量。
- 是否符合类型化目标的基本语义。

它的目标是帮助 VISTA 选择和修复候选。

### 7.2 官方 评分器

`src/parallel_score.py` 和 `src/bridge.py` 调用 NetArena 官方评估器。它负责最终基准测试指标：

- 正确性。
- 安全性。
- 延迟。

这两级验证的区别是：

- 内部验证器是规划和修复的一部分。
- 官方评分器是实验评估的一部分。

---

## 8. 当前实现的实际创新点和可改进点

### 8.1 已经落地的创新点

- 类型化目标 编译：`src/intent.py`。
- 信念快照 注入：`src/belief.py`。
- 案例记忆检索：`src/retrieval.py`。
- 多假设生成：`n_hyp`。
- 带证明的验证：`src/verifier.py`。
- 证明引导修复：`src/planner.py`、`src/run_vista.py`、`src/run_all_gen.py`。
- 输出格式约束归一化：`src/polish.py`。
- 对齐官方基准测试的评估：`src/bridge.py`、`src/parallel_score.py`。
- 带意图提示的 SFT：`src/sft_train.py`。

### 8.2 仍是方法论但未完全工程化的部分

- 时序类型化信念图中的多源遥测、工单、变更历史目前没有真实数据接入。
- 信念编码器目前没有训练时序/图神经网络编码器。
- 假设后验目前不是显式概率分布，而是候选程序 + 证明置信度。
- 信息增益公式没有显式计算。
- 风险目前主要由不变量检查和证明置信度体现。
- 反事实模拟目前是单个 MALT 图副本执行，不是多世界 `W_t`。

这些不是缺陷，而是 基准测试 落地版本和完整研究框架之间的边界。当前代码优先实现了 MALT 上可复现、可量化、能提升 安全性 的闭环。

---

## 9. 结果文件和实验产物

虽然这些不是源码，但理解项目很重要：

- `data/train.jsonl`：训练集划分。
- `data/test.jsonl`：测试集划分。
- `outputs/raw_*.jsonl`：大语言模型生成的原始结果和抽取代码。
- `outputs/*.jsonl`：官方 评分器 评分后的结果。
- `outputs/*.summary.json`：每个方法的汇总指标。
- `outputs/RESULTS.md`：总表和逐标签表。
- `outputs/SUMMARY.md`：完整实验总结。
- `logs/*.out`、`logs/run_pipeline.log`：运行日志。

当前 `outputs/SUMMARY.md` 中的核心结果显示，VISTA-base 和 VISTA-SFT 在 1000 条留出 MALT 样本上正确性达到 1.000，安全性达到 0.709，高于语义少样本的 0.589 安全性。

---

## 10. 核心实验日志解读

项目里有很多日志文件，其中最核心的是三类：全流程日志、VISTA 生成日志、VISTA 评分日志。它们分别对应“实验有没有完整跑完”、“VISTA 内部验证器/修复发生了什么”、“最终官方评分结果是多少”。

### 10.1 核心日志一：全流程日志

推荐优先看归档目录中的：

```text
/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/logs/full_compare.log
```

这个日志是完整对比实验的主线日志。它记录了从生成、评分、聚合到归档的完整过程。和 `/home/ubuntu/vista/logs/run_pipeline.log` 相比，`full_compare.log` 是当前 `outputs/SUMMARY.md` 对应的规范运行；`run_pipeline.log` 中还保留了更早的中断/恢复痕迹，因此做论文或报告时优先引用 `runs/full_compare_.../logs/full_compare.log`。

关键字段怎么看：

- `PHASE=...`：表示当前实验阶段。例如生成、评分、聚合、归档输出。
- `END_UTC=2026-05-05T08:26:10Z`：完整实验结束时间。
- `DONE`：说明全流程结束并归档成功。
- `Wrote: .../outputs/RESULTS.md` 和 `RESULTS.csv`：说明聚合结果表已生成。

这份日志最重要的意义是证明实验不是只跑了某个单独模型，而是完整比较了 零样本、少样本、语义少样本、SFT-only、VISTA-base、VISTA-SFT，并最终生成统一结果。

如果只想看摘要，不需要翻大日志，直接看：

```text
/home/ubuntu/vista/runs/full_compare_20260505_053226_UTC/SUMMARY.md
/home/ubuntu/vista/outputs/SUMMARY.md
```

这两份摘要记录了运行编号、时间、模型、测试集规模、整体结果、按任务范围划分的快照、VISTA 生成统计和完整产物路径。

### 10.2 核心日志二：VISTA 生成日志

VISTA 生成过程最直接的日志是：

```text
/home/ubuntu/vista/logs/vista_base_full.out
/home/ubuntu/vista/logs/vista_sft_full.out
```

这两份日志分别对应基础模型上的 VISTA 和 SFT 适配器上的 VISTA。它们记录每个分块的内部验证器通过情况和修复次数，例如：

```text
[gen:vista_base] chunk 125/125 init_pass=8/8 +repairs=0 running pass=947/1000
[gen:vista_base] DONE  init_pass=947/1000 repair_iters=159
```

字段含义：

- `chunk 125/125`：1000 条测试样本，分块大小为 8，所以一共 125 个分块。
- `init_pass=8/8`：当前分块中，初始多假设采样后有 8 条直接通过内部验证器。
- `+repairs=0`：当前分块中修复轮次生成次数为 0。
- `running pass=947/1000`：到当前分块为止，初始候选直接通过验证器的累计条数。
- `DONE init_pass=947/1000 repair_iters=159`：整个 VISTA-base 生成完成；1000 条里 947 条无需修复，直接通过内部证明检查门，剩余失败样本触发了 159 次修复生成。

这个日志对应我们方法里的“多假设推理 + 证明引导修复”：

1. 每条查询先编译目标、构造信念快照、检索案例。
2. 大语言模型一次采样 `n_hyp=5` 个候选程序。
3. 每个候选进入 `verify()`。
4. 如果没有候选通过，就使用 `Proof.repair_hint` 继续修复。
5. 日志中的 `init_pass` 和 `repair_iters` 就是这个闭环的运行痕迹。

从日志可以看到，VISTA 不是简单提示词基线。它内部先做验证器筛选，再做修复。`init_pass=947/1000` 说明绝大多数任务在检索 + 类型化目标 + 信念快照的约束下已经能直接生成合格候选；`repair_iters=159` 说明仍有一部分复杂或不稳定任务需要证明引导修复。

注意：`outputs/SUMMARY.md` 中记录的最终 生成统计 为：

| 方法 | 初始通过数 | 修复轮次数 |
|---|---:|---:|
| 基础模型上的 VISTA | 947/1000 | 159 |
| SFT 适配器上的 VISTA | 947/1000 | 159 |

如果某个临时日志和 摘要 数字不一致，优先以 `outputs/SUMMARY.md` 和 `runs/full_compare_20260505_053226_UTC/SUMMARY.md` 为准，因为它们对应最终归档后的 完整对比实验。

### 10.3 核心日志三：VISTA 官方评分日志

评分日志主要看：

```text
/home/ubuntu/vista/logs/score_vista.out
```

它记录 `score_all` 或 `parallel_score` 调用 NetArena 官方评估器的过程。该阶段和内部验证器不同：内部验证器是 VISTA 生成阶段的快速证明检查门；官方评分器是最终基准测试指标来源。

典型尾部会出现：

```text
[score:vista_sft] 1000 records in 579.9s = 0.58s/q
[score:vista_sft] {'方法': 'vista_sft', 'n': 1000, '正确性': 1.0, '安全性': 0.709, ...}
```

字段含义：

- `1000 records`：评分样本数，必须等于测试集的 1000。
- `0.58s/q`：官方评估器平均每条评分耗时。这里主要耗时来自大图上的图同构检查和安全性检查。
- `correctness`：NetArena 标准答案等价性结果。
- `safety`：NetArena 安全检查器结果。
- `avg_latency_s`：生成阶段记录的平均推理延迟，不是评分器的耗时。

评分日志里还会出现大量逐查询输出：

- `Verifier results: True/False`
- `Ground truth verifier results: True/False`
- `LLM code result`
- `Ground truth result`

这些细节用于定位 失败模式。常见失败包括：

- `Invalid hierarchy`：例如把 `EK_PORT` 直接挂到 `EK_CONTROL_DOMAIN` 或 `EK_AGG_BLOCK` 下，违反 MALT 层级约束。
- `Found 16 isolated nodes`：删除某个 switch 后，其 ports 变成孤立节点，违反 `no_isolated_nodes`。
- 排序/列表/计数的 `data` 与标准答案不一致：正确性失败，但不一定安全性失败。

最终的权威数字来自 `outputs/*.summary.json` 和 `outputs/RESULTS.md`：

```text
/home/ubuntu/vista/outputs/vista_base.summary.json
/home/ubuntu/vista/outputs/vista_sft.summary.json
/home/ubuntu/vista/outputs/RESULTS.md
```

当前最终结果为：

| 方法 | 数量 | 正确性 | 安全性 | 平均延迟 |
|---|---:|---:|---:|---:|
| VISTA-base | 1000 | 1.000 | 0.709 | 2.460s |
| VISTA-SFT | 1000 | 1.000 | 0.709 | 2.504s |
| 语义少样本基线 | 1000 | 1.000 | 0.589 | 0.271s |
| SFT-only 基线 | 1000 | 1.000 | 0.000 | 0.236s |

这里最关键的结论是：VISTA 在保持正确性 1.000 的同时，把最佳提示词基线的安全性从 0.589 提升到 0.709，绝对提升 0.120；同时说明单独 SFT 并不能保证安全，必须叠加 VISTA 的类型化目标、信念快照、案例检索、带证明的验证和修复。

---

## 11. 如何运行实验

实验运行分为 冒烟测试、单模块、完整全流程、恢复运行和只评分几种模式。

### 11.1 环境前提

运行前需要确认：

- NetArena 在 `/home/ubuntu/NetArena`。
- MALT 应用在 `/home/ubuntu/NetArena/app-malt`。
- 基础模型在 `/home/ubuntu/vista/models/Qwen2.5-Coder-7B-Instruct`。
- 嵌入向量模型在 `/home/ubuntu/vista/models/bge-base-en-v1.5`。
- 如果要跑 SFT/VISTA-SFT，LoRA 适配器 在 `/home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final`，或可以重新训练得到。
- GPU 环境可加载 vLLM。

所有命令默认从项目根目录运行：

```bash
cd /home/ubuntu/vista
```

### 11.2 准备数据

```bash
python -m src.data
```

输出：

```text
data/train.jsonl
data/test.jsonl
```

默认每个标签取 400 条训练样本、100 条测试样本，共 4000 条训练样本、1000 条测试样本。

### 11.3 快速 冒烟运行

如果只是检查 流水线 是否能跑，不想跑完整 1000 条，可以给脚本传 limit：

```bash
bash scripts/run_full.sh 50
```

这会只取前 50 条测试样本。适合检查：

- 数据读取是否正常。
- vLLM 是否能加载。
- 检索器缓存是否能建立。
- 评分器是否能调用 NetArena 官方评估器。
- `outputs/` 和 `logs/` 目录是否能写出。

### 11.4 完整实验

推荐使用：

```bash
bash scripts/run_full.sh
```

该脚本会执行：

1. `python -m src.data`
2. `python -m src.run_all_gen`：基础模型基线 + VISTA-base 生成
3. `python -m src.score_all --num-workers 12`
4. `python -m src.sft_train`
5. `python -m src.run_all_gen --lora ... --baselines sft_only vista_sft`
6. `python -m src.score_all --num-workers 12`
7. `python -m src.aggregate`

完整运行会写：

```text
outputs/raw_*.jsonl
outputs/baseline_*.jsonl
outputs/vista_*.jsonl
outputs/*.summary.json
outputs/RESULTS.md
outputs/RESULTS.csv
logs/run_pipeline.log
```

### 11.5 只跑 VISTA-base

```bash
python -m src.run_vista \
  --chunk-size 8 \
  --n-hyp 5 \
  --max-repair 3 \
  --num-workers 12
```

如果只生成不评分：

```bash
python -m src.run_vista \
  --phase gen \
  --chunk-size 8 \
  --n-hyp 5 \
  --max-repair 3
```

如果 原始 已经生成，只补评分：

```bash
python -m src.run_vista \
  --phase score \
  --num-workers 12
```

### 11.6 只跑 VISTA-SFT

```bash
python -m src.run_vista \
  --lora /home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final \
  --chunk-size 8 \
  --n-hyp 5 \
  --max-repair 3 \
  --num-workers 12
```

### 11.7 只跑 基线

例如 零样本：

```bash
python -m src.run_baseline --name zeroshot_base --batch-size 16 --num-workers 12
```

例如 语义少样本：

```bash
python -m src.run_baseline --name fewshot_semantic --batch-size 16 --num-workers 12
```

例如 SFT-only：

```bash
python -m src.run_baseline \
  --name sft_only \
  --lora /home/ubuntu/vista/models/sft-qwen-coder-7b-malt/final \
  --batch-size 16 \
  --num-workers 12
```

### 11.8 只评分已有 原始 文件

如果已经有 `outputs/raw_*.jsonl`，可以直接跑：

```bash
python -m src.score_all --num-workers 12
```

只评分指定方法：

```bash
python -m src.score_all --num-workers 12 --names vista_base vista_sft
```

### 11.9 重新聚合结果表

```bash
python -m src.aggregate
```

这会重新生成：

```text
outputs/RESULTS.md
outputs/RESULTS.csv
```

### 11.10 从 SFT 阶段恢复

如果 base 生成 已经完成，只需要重跑 SFT 及其后续：

```bash
bash scripts/resume_from_sft.sh
```

### 11.11 日志查看建议

全流程运行时可以看：

```bash
tail -F /home/ubuntu/vista/logs/run_pipeline.log
```

VISTA 生成 单独日志可看：

```bash
tail -F /home/ubuntu/vista/logs/vista_base_full.out
tail -F /home/ubuntu/vista/logs/vista_sft_full.out
```

评分阶段可看：

```bash
tail -F /home/ubuntu/vista/logs/score_vista.out
```

### 11.12 运行完成后检查

完整实验完成后，至少检查：

```bash
wc -l outputs/raw_vista_base.jsonl
wc -l outputs/raw_vista_sft.jsonl
wc -l outputs/vista_base.jsonl
wc -l outputs/vista_sft.jsonl
cat outputs/vista_base.summary.json
cat outputs/vista_sft.summary.json
```

正常完整运行中，每个原始/已评分 JSONL 应该有 1000 行，摘要中 `n` 应该为 1000。

---

## 12. 用一句话总结代码方案

这个项目把网络运维任务从“一次性代码生成”改造成了一个更可靠的流程：

```text
先确定结构化目标，
再查询和压缩当前网络状态，
再检索历史程序案例，
再生成多个候选假设，
再用证明验证器做执行前模拟和安全验证，
失败则带着诊断修复，
最后用官方评估器做严格评分。
```

这种设计的价值在于，大语言模型不再是唯一的决策来源。大语言模型负责提出候选，类型化目标、信念图、案例记忆和验证器共同约束候选，使系统在网络图操作这种高安全要求任务上更稳定、更可解释、更容易调试。
