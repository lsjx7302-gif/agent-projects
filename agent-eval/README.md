# Agent 工具调用评测

一个面向 Function Calling Agent 的可复现评测工程：24 条诊断任务、4 个工具，覆盖单工具、多工具、多轮、工具报错、无答案；输出任务成功率、工具选择 / 参数准确率、RAGAS ToolCallAccuracy、平均轮数与 token。

**技术栈**：Python · OpenAI-compatible Function Calling · RAGAS · Qwen3.8-Flash  
**作者**：石家旭

## 1. 背景

Agent 的工具调用质量不能只看最终答案：工具选错、参数错、漏调、多调、遇到错误编造，都会在上线后暴露。本项目用一个小而可控的任务集，把这些问题拆开量化，并保留完整的工具调用 trace，便于定位 badcase。

## 2. 架构

```
question
  → Agent loop（OpenAI 兼容接口，max 6 steps）
      → LLM 返回 tool_calls
      → 执行工具（错误作为 observation 返回，不抛异常）
      → 回填 tool 消息，继续下一轮
  → 无 tool_calls 时输出最终答案
```

每次运行记录：工具名、参数、结果、LLM 轮数、时延、token 用量。工具错误一律返回 `ERROR: ...`，由模型决定重试 / 换工具 / 如实告知。

## 3. 工具

| 工具 | 作用 | 错误路径 |
|---|---|---|
| `get_weather` | 查询城市天气 | 未知城市 |
| `search_rules` | 检索电力现货规则 | 无匹配条款 |
| `calculate` | 四则运算 | 除零 / 非法表达式 |
| `get_price_summary` | 查询某天 96 点电价摘要 | 无数据日期 |

## 4. 任务集（24 条）

| 类别 | 数量 | 测什么 |
|---|---|---|
| single_tool | 8 | 正确选一个工具、参数正确、答案落地 |
| multi_tool | 6 | 两个以上工具协作（天气 + 计算、规则 + 计算、多日价格） |
| multi_round | 4 | 多轮依赖：先查 A，再查 B，再算差值 / 百分比 |
| tool_error | 3 | 未知城市、除零、无数据日期；要求如实说明，不编造 |
| no_answer | 3 | 订票 / 股市 / 公司预算；要求不调无关工具，直接说明无法回答 |

数据：`data.py` 用固定 seed 生成 3 天 × 96 点电价、5 个城市天气、6 条合成规则。任务与期望值由 `make_tasks.py` 生成，`dry_run.py` 可在不调用 LLM 的情况下校验任务集。

## 5. 指标

- **任务成功率**：答案中的数字（带容差）与关键词全部命中期望。
- **工具选择准确率（严格）**：预测工具序列与参考序列完全一致（多调 / 少调都扣分）。
- **工具 F1 / 必要工具召回**：按工具名做 precision / recall / F1，能区分「多调可选工具」和「漏调必要工具」。
- **参数准确率**：城市 / 日期精确匹配、算式数值等价、检索 query 包含关系。
- **RAGAS ToolCallAccuracy**：ragas 0.3.1 的多轮指标，严格比较工具名和参数；检索 query 措辞不同会显著拉低分数，因此与自建指标并列看。

## 6. 结果

| 指标 | 数值 |
|---|---|
| 任务成功率 | **100.0%**（24/24） |
| 工具选择准确率（严格序列） | 91.7% |
| 工具调用 F1 | **0.95** |
| 必要工具召回 | 95.8% |
| 参数准确率 | 87.5% |
| RAGAS ToolCallAccuracy（严格参数） | 0.6667 |
| 平均工具调用数 | 1.38 |
| 平均 LLM 轮数 | 2.25 |
| 平均时延 | 3.31 s |
| 总 token | 45,181 |

> 任务成功率 100% 只代表这 24 条自建诊断任务；不代表通用 Agent 能力。

## 7. Badcase 与迭代

1. **第一版 21/24**：3 条是模型用更少工具完成了任务（口算加法、自己识别非法日期、规则里已有的 96 直接引用）。→ 调整任务，让工具成为必要而不是可选。
2. **T24 公司预算**：模型误调 `search_rules`。→ system prompt 增加「如果问题与工具能力无关，不要调用工具」，修复为 0 次工具调用。
3. **当前 2 条偏差**：T16 同一算式调了两次 `calculate`（冗余，答案正确）；T19 火星天气没调 `get_weather`，直接用模型知识回答（答案正确，但缺少工具 grounding）。
4. **RAGAS 66.7% vs 自建 F1 95%**：RAGAS 严格比较 `search_rules` 的 query 字符串，措辞不同就判错；这说明「指标口径」本身要选对，不能只看一个数。

## 8. 我的工作

- 设计 24 条诊断任务、参考工具序列、期望数字 / 关键词与容差。
- 实现 Agent 主循环、4 个工具、错误处理与完整 trace 记录。
- 实现任务成功率、工具选择 / F1、参数准确率等指标，并接入 RAGAS ToolCallAccuracy。
- 基于 badcase 迭代任务集与 system prompt；用 `dry_run.py` 先验证任务本身再花 LLM 调用。
- 全部代码与文档独立实现。

## 9. 快速开始

```bash
cd agent-eval
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# 设置 DASHSCOPE_API_KEY 后：
.venv/Scripts/python.exe make_tasks.py
.venv/Scripts/python.exe dry_run.py
.venv/Scripts/python.exe run_eval.py --model qwen3.8-flash
```

> ragas 0.3.1 需要 `langchain-community<0.4`（0.4 移除了 `chat_models.vertexai`）和 Pillow，`requirements.txt` 已固定。

## 10. 项目结构

```
agent-eval/
├── agent.py          # Agent 主循环
├── tools.py          # 4 个工具 + function schema
├── data.py           # 合成天气 / 规则 / 电价
├── make_tasks.py     # 生成 tasks.json
├── dry_run.py        # 不调 LLM，验证任务集
├── run_eval.py       # 评测 + RAGAS + 汇总
└── results/          # 结果与 summary
```

## 11. 局限

- 自建 24 条任务、合成数据、单模型（qwen3.8-flash）；不是公开 benchmark。
- 答案判定用数字容差 + 关键词，不是 LLM judge；对开放式回答不适用。
- RAGAS 只用了 `ToolCallAccuracy`，没有跑 faithfulness / context recall；后续可接 RAG 评测。
