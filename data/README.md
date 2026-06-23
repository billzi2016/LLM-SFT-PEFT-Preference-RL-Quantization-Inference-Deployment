# 数据样例说明

本目录只放少量真实可用样例，不放大规模数据集。

## 文件

- `sample_sft.jsonl`：用于 SFT、LoRA、QLoRA。
- `sample_preference_dpo.jsonl`：用于 DPO。
- `sample_prompts_rl.jsonl`：用于 PPO、GRPO、奖励函数演示。

## 要求

- 每一行必须是一个完整 JSON 对象。
- 不允许跨行 JSON。
- 不允许使用 `TODO`、`略`、`自行补充` 这类占位内容。
- 字段必须能被后续 notebook 直接读取和处理。
