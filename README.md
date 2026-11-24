# Agent 项目（石家旭）

两个可复现的 Agent 项目：

## 1. [agent-eval](./agent-eval) — Agent 工具调用评测

- 24 条诊断任务集（单工具 / 多工具 / 多轮 / 工具报错 / 无答案），覆盖天气、规则检索、计算、电价 4 类工具。
- 结果：任务成功率 **100%（24/24）**，工具调用 F1 **95.0%**，必要工具召回 95.8%，参数准确率 87.5%，平均 2.25 轮 / 3.3s。
- 模型：qwen3.8-flash；评测代码 `run_eval.py`，任务集 `tasks.json`，结果在 `results/`。
- 运行方式见 [agent-eval/README.md](./agent-eval/README.md)。

## 2. [tradingagents-power](./tradingagents-power) — 电力现货多智能体复盘

- 参考 TradingAgents 的角色分工与条件边，独立实现电力现货场景（不是 fork）。
- 4 分析师并行 → 多空辩论 → 研究经理 → 交易员 → 3 风控 → PM；输出 96 点价格曲线。
- 硬约束：逐点 ±30%、分段 ±20%、全天 ±15%、[40,650]、跳变 ≤20；3 个交易日 **0 违规**。
- 单次 11 次 LLM 调用 / ~132s；3 个交易日消融，含 always-flat 基线。
- 运行方式见 [tradingagents-power/README.md](./tradingagents-power/README.md)。

## 说明

- 所有数据为合成数据，不使用公司数据。
- 仓库不包含任何 API key；运行前请设置 `DASHSCOPE_API_KEY`。
- `agent-eval` 使用 ragas 0.3.1，需 `langchain-community<0.4` 与 Pillow（见 requirements.txt）。
