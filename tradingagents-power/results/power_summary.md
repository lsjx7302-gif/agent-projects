# 电力现货多智能体复盘（二次开发）结果

- 模型：`qwen3.8-flash`
- 目标日：2026-11-23, 2026-11-24, 2026-11-25
- 数据：合成 96 点负荷/风光/检修/电价

## 总体

| 指标 | 多 Agent（分析师 + 辩论 + 风控） | 单 Agent | 不变化基线 |
|---|---|---|---|
| 分段方向正确 | 3/9 | 3/9 | - |
| MAE（元/MWh） | 11.14 | 10.21 | 7.04 |
| 平均 LLM 调用 | 11.0 | 1.0 | - |
| 平均时延（s） | 132.4 | 71.7 | - |

## 每日明细

### 2026-11-23
- 多 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'flat'} / 实际 {'peak': 'flat', 'valley': 'down', 'flat': 'flat'}；正确 2/3；MAE 9.7；风险等级 medium
- 单 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'down'} / 实际 {'peak': 'flat', 'valley': 'down', 'flat': 'flat'}；正确 1/3；MAE 11.12
- 约束违规：{'point_violation': 0.0, 'segment_violation': 0.0, 'daily_violation': 0.0, 'jump_violation': 0.0, 'range_violation': 0.0}

### 2026-11-24
- 多 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'flat'} / 实际 {'peak': 'flat', 'valley': 'up', 'flat': 'flat'}；正确 1/3；MAE 10.43；风险等级 medium
- 单 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'flat'} / 实际 {'peak': 'flat', 'valley': 'up', 'flat': 'flat'}；正确 1/3；MAE 10.17
- 约束违规：{'point_violation': 0.0, 'segment_violation': 0.0, 'daily_violation': 0.0, 'jump_violation': 0.0, 'range_violation': 0.0}

### 2026-11-25
- 多 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'down'} / 实际 {'peak': 'flat', 'valley': 'flat', 'flat': 'flat'}；正确 0/3；MAE 13.29；风险等级 medium
- 单 Agent：方向 {'peak': 'up', 'valley': 'down', 'flat': 'flat'} / 实际 {'peak': 'flat', 'valley': 'flat', 'flat': 'flat'}；正确 1/3；MAE 9.34
- 约束违规：{'point_violation': 0.0, 'segment_violation': 0.0, 'daily_violation': 0.0, 'jump_violation': 0.0, 'range_violation': 0.0}
