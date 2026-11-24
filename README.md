# Agent 项目

作者：石家旭 · GitHub：[@lsjx7302-gif](https://github.com/lsjx7302-gif)

两个 Agent 工程实践：

## 1. [agent-eval](./agent-eval) — Agent 工具调用评测

面向 Function Calling Agent 的可复现评测工程：24 条诊断任务、4 个工具，覆盖单工具 / 多工具 / 多轮 / 工具报错 / 无答案。

- 任务成功率 **100%（24/24）**
- 工具调用 F1 **95.0%**，必要工具召回 95.8%，参数准确率 87.5%
- 平均 2.25 轮 LLM / 1.38 次工具调用 / 3.31s
- 接入 RAGAS ToolCallAccuracy，并与自建指标对照分析口径差异

## 2. [tradingagents-power](./tradingagents-power) — 电力现货多智能体复盘系统

负荷 / 新能源 / 检修 / 价差 4 类分析师并行 → 多空辩论 → 研究经理 → 交易员 → 3 风控 → PM，输出 96 点价格曲线。

- 逐点 ±30% / 分段 ±20% / 全天 ±15% / [40,650] / 跳变 ≤20，3 个交易日 **0 违规**
- 单次 11 次 LLM 调用 / 约 132s
- 3 个交易日消融（多 Agent / 单 Agent / 不变化 / always-flat），定位模型过度方向性 badcase

## 数据与安全

- 全部为合成数据，不使用公司数据。
- 仓库不包含任何 API key；运行前请设置 `DASHSCOPE_API_KEY`。
- `agent-eval` 使用 ragas 0.3.1，需 `langchain-community<0.4` 与 Pillow（见 requirements.txt）。
