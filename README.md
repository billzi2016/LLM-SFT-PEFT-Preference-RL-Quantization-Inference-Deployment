# LLM-SFT

这个仓库用于整理两套开源权重大模型的训练、量化和部署流程：

- `Qwen/Qwen2.5-7B-Instruct`
- `openai/gpt-oss-20b`

内容优先使用 notebook。Markdown 只保留入口、规划和索引；真实讲解、参数说明、可运行代码、数据格式说明都尽量放到 `.ipynb`。

## 推荐阅读顺序

1. `prd.md`：查看整体规划、文件拆分、方法排名和验收标准。
2. `notebooks/shared/00_data_formats.ipynb`：理解 SFT、DPO、RL 数据格式。
3. `notebooks/qwen2_5_7b/01_model_overview.ipynb`：理解 Qwen2.5-7B 的定位和结构。
4. `notebooks/gpt_oss_20b/01_model_overview.ipynb`：理解 gpt-oss-20b 的定位、MoE、Harmony、MXFP4。

## 当前已创建内容

- `data/sample_sft.jsonl`：SFT、LoRA、QLoRA 可用样例。
- `data/sample_preference_dpo.jsonl`：DPO 偏好数据样例。
- `data/sample_prompts_rl.jsonl`：PPO/GRPO 可验证奖励样例。
- `notebooks/shared/00_data_formats.ipynb`：数据格式说明。
- `notebooks/qwen2_5_7b/01_model_overview.ipynb`：Qwen2.5-7B 模型介绍。
- `notebooks/gpt_oss_20b/01_model_overview.ipynb`：gpt-oss-20b 模型介绍。

## 基本原则

- 样例数据必须是真实 JSONL，可被 `datasets.load_dataset("json", data_files=...)` 读取。
- notebook 中的代码必须是真实可执行代码，不能写伪代码。
- 同类内容聚合到一个 notebook 中。
- Qwen2.5-7B 和 gpt-oss-20b 分成两套 notebook。
