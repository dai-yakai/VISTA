# VISTA 三个主实验重跑审计

运行目录：`/home/ubuntu/vista/runs/all3_rerun_20260513_185405_UTC`

本次重跑的目标是按真正主实验流程重新跑 MALT、Route、K8s 三个基准，并重点检查结果是否可信，尤其是排查 Route 之前出现 90% 以上成功率是否由实验设置、流程或隐藏捷径造成。

## 总结果

| 基准 | 样本数 | 主指标 | 安全率 | 平均步数/延迟 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| MALT | 1000 | 正确率 99.2% | 86.9% | 3.520 秒 | 完整生成并重新评分，结果可信 |
| Route | 150 | 成功率 36.7% | 98.7% | 9.047 步 | 之前 90%+ 不采信，本次结果可信 |
| K8s | 150 | 成功率 29.3% | 88.0% | 8.513 步 | 完整主实验结果可信 |

## 运行设置

本次使用本地模型：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/vista/models/Qwen2.5-Coder-7B-Instruct \
  --served-model-name qwen-local \
  --host 127.0.0.1 --port 8008 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Route 与 K8s 使用同一个主智能体入口：

```bash
python /home/ubuntu/vista_agents/main_a2a_agent.py --mode route --port 8000
python /home/ubuntu/vista_agents/main_a2a_agent.py --mode k8s --port 8000
```

智能体卡片中明确记录为通用 VISTA 主智能体，没有应用专用标准答案表：`No app-specific canonical tables; uses only prompt/history and generic validation.`

## MALT

MALT 输出：

- 原始生成：`outputs/malt/raw_vista_base.jsonl`，1000 行。
- 评分结果：`outputs/malt/vista_base.jsonl`，1000 行。
- 汇总文件：`outputs/malt/vista_base.summary.json`。

日志显示主生成阶段参数为：

- 样本数：1000。
- 候选假设数：5。
- 最大修复轮数：3。
- 初始验证通过：869/1000。
- 总修复尝试：393。

最终评分结果：

```json
{
  "method": "vista_base",
  "n": 1000,
  "correctness": 0.992,
  "safety": 0.869,
  "avg_latency_s": 3.519984071264982
}
```

审计说明：MALT 的生成阶段已经产出完整 1000 条原始记录。为了释放显存给 Route/K8s 的在线智能体服务，原来的全流程进程在原始记录完成后停止，随后使用同一个原始文件做确定性评分重启。这个步骤没有重新生成答案，只是对 `raw_vista_base.jsonl` 进行评分，因此不会改变模型行为样本。

## Route

Route 主实验设置：

- 配置文件：`config/route_full.toml`。
- 固定查询文件原路径：`/home/ubuntu/vista/runs/route_full_compare_20260505_120708_UTC/config/route_full_150_queries.json`。
- 为便于打包复现，已复制到：`config/route_full_150_queries.used.json`。
- 查询文件哈希：`4ee206f50f3368382c0c4431c3412ec422334bb08215fbd0c82d353803c376ec`。
- 查询数：150。
- 每类故障：10 条。
- 最大交互轮数：10。
- 拓扑：2 个交换机，每个子网 1 个主机。

Route 汇总：

| 故障类型 | 成功率 | 安全率 | 平均步数 |
| --- | ---: | ---: | ---: |
| disable_routing | 100.0% | 100.0% | 5.4 |
| remove_ip | 10.0% | 100.0% | 9.1 |
| drop_traffic_to_from_subnet | 100.0% | 100.0% | 9.0 |
| disable_interface | 70.0% | 100.0% | 6.5 |
| wrong_routing_table | 0.0% | 100.0% | 10.0 |
| disable_routing+disable_interface | 60.0% | 100.0% | 9.3 |
| disable_routing+drop_traffic_to_from_subnet | 100.0% | 100.0% | 9.2 |
| disable_routing+remove_ip | 0.0% | 100.0% | 10.0 |
| disable_routing+wrong_routing_table | 0.0% | 100.0% | 10.0 |
| disable_interface+drop_traffic_to_from_subnet | 40.0% | 100.0% | 9.6 |
| disable_interface+remove_ip | 10.0% | 100.0% | 9.5 |
| disable_interface+wrong_routing_table | 30.0% | 80.0% | 8.5 |
| remove_ip+drop_traffic_to_from_subnet | 30.0% | 100.0% | 9.6 |
| remove_ip+wrong_routing_table | 0.0% | 100.0% | 10.0 |
| drop_traffic_to_from_subnet+wrong_routing_table | 0.0% | 100.0% | 10.0 |

审计结论：

- 本次 Route 不是 90%+，而是 36.7%。这说明之前的 90%+ 结果不应采信。
- 日志显示共 150 条查询、150 个逐题结果、15 个类型汇总，和主实验配置一致。
- 日志里能看到大量真实失败，例如 `wrong_routing_table`、`remove_ip` 相关类型常常跑满 10 步仍未修复。
- 最后一批日志中，模型连续查看路由、规则、接口、地址、iptables、队列等信息，仍然可能失败，说明不是直接读取标准答案。
- `app-route` 与官方代码对比只多了运行时产生的 `nodes` 目录，没有修改评测逻辑。

## K8s

K8s 主实验设置：

- 配置文件：`config/k8s_full.toml`。
- 生成故障文件：`config/k8s_full_errors.json`。
- 故障文件哈希：`2847942248fcbc65e0ba24965e4d52ff0bcaa95ca1abceb291785c6de6c9cd46`。
- 查询数：150。
- 每类故障：10 条。
- 最大交互轮数：10。
- 输出汇总：`outputs/k8s/vista-main-k8s-full_PromptType.ZEROSHOT_BASE/20260514-043921/test_results_summary.json`。

K8s 汇总：

| 故障类型 | 成功数/10 | 安全数/10 | 平均步数 |
| --- | ---: | ---: | ---: |
| add_egress | 0 | 10 | 10.0 |
| add_ingress | 5 | 10 | 6.7 |
| add_ingress+add_egress | 0 | 8 | 10.0 |
| add_ingress+change_port | 3 | 9 | 9.2 |
| add_ingress+change_protocol | 4 | 9 | 8.7 |
| change_port | 8 | 10 | 5.0 |
| change_port+add_egress | 0 | 8 | 10.0 |
| change_port+change_protocol | 5 | 9 | 8.0 |
| change_protocol | 8 | 10 | 4.7 |
| change_protocol+add_egress | 0 | 10 | 10.0 |
| remove_ingress | 3 | 7 | 8.0 |
| remove_ingress+add_egress | 0 | 6 | 10.0 |
| remove_ingress+add_ingress | 1 | 8 | 9.3 |
| remove_ingress+change_port | 2 | 8 | 9.6 |
| remove_ingress+change_protocol | 5 | 10 | 8.5 |

审计结论：

- K8s 完整产出 150 个逐题结果和 1 个官方汇总文件。
- 日志显示真实部署、注入网络策略错误、逐轮 `kubectl` 命令、逐轮连通性检查，没有只跑少量样本冒充全集。
- 日志里存在大量跑满 10 步的失败，特别是 `add_egress` 和若干组合故障，因此结果没有虚高。
- K8s 本地代码相对官方代码有环境适配：等待目标容器、使用本地 `busybox:1.36.1-local`、更长部署等待、部署前清理旧策略。这些是为了让本地集群稳定运行，不包含标准答案逻辑。
- K8s 的 `text_utils.py` 已恢复为官方提取逻辑，仅剩文件末尾换行差异。

## 可信度检查

本次检查重点：

1. 是否完整跑完主实验样本数。
2. 是否沿用主实验交互流程。
3. 是否用了官方评测器生成的逐题结果和汇总。
4. 是否存在标准答案表、直接读答案、只跑简单子集、改评测器等会让结果虚高的情况。
5. Route 结果是否仍异常高。

检查结果：

- MALT：1000 条原始生成、1000 条评分结果完整。
- Route：150 条逐题结果完整，成功率 36.7%，不再异常高。
- K8s：150 条逐题结果完整，成功率 29.3%，日志中失败样本充分。
- 未发现让结果虚高的捷径。

因此，本次重跑结果可以作为当前可信版本保存；尤其是 Route，之前 90%+ 的结果应作废，以本次 36.7% 为准。
