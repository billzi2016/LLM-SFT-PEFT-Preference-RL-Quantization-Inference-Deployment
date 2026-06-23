# 大模型训练与部署内容规划

## 1. 文档目标

本文件用于规划当前仓库后续要补齐的全部内容。目标不是只给几段代码，而是形成一套能从零跑通的大模型训练、量化、推理、部署材料。

最终内容需要让第一次接触大模型微调的人也能看懂：

- 为什么要做这个步骤。
- 这个步骤输入什么。
- 这个步骤输出什么。
- 关键参数分别控制什么。
- 什么场景该选哪条路线。
- 同一模型在训练、量化、部署之间如何衔接。

本仓库主要覆盖两套模型：

- `Qwen/Qwen2.5-7B-Instruct`
- `openai/gpt-oss-20b`

其中 Qwen2.5-7B 作为 7B dense 模型代表，适合先跑通完整流程；gpt-oss-20b 作为 MoE 开放权重模型代表，重点体现 Harmony 模板、MXFP4、reasoning effort、vLLM/Ollama/Unsloth 等差异。

## 2. 总体范围

### 2.1 训练方法

需要覆盖：

- SFT：监督微调，用已有输入输出样本训练模型按照指定方式回答。
- LoRA：只训练低秩适配器，减少显存和训练成本。
- QLoRA：用 4bit 加载基座模型，再训练 LoRA 适配器，进一步降低显存。
- DPO：使用 chosen/rejected 偏好数据，让模型更偏向优质回答。
- PPO：使用奖励模型或奖励函数，对模型输出进行强化优化。
- GRPO：使用组内相对奖励，减少 PPO 中 value model 的额外成本，适合可验证奖励任务。

### 2.2 量化方法

需要覆盖：

- bitsandbytes 8bit：常用于低显存加载和训练前准备。
- bitsandbytes 4bit：QLoRA 常用加载方式。
- GGUF：llama.cpp、Ollama、LM Studio 常用格式。
- GPTQ：常见离线权重量化格式，偏推理使用。
- AWQ：面向推理的激活感知权重量化，常用于 vLLM、Transformers、AutoAWQ 生态。
- EXL2：ExLlamaV2 常用格式，适合单机高效推理。
- FP8：服务端推理常用低精度方案，常见于新 GPU 和 vLLM 场景。
- MXFP4：gpt-oss 系列重点格式，OpenAI gpt-oss 权重中大量 MoE 权重使用该压缩方式。

### 2.3 部署方式

需要覆盖：

- Transformers 本地推理：最基础、最容易理解，适合验证模型和 adapter 是否可用。
- vLLM 部署：高吞吐 OpenAI-compatible API 服务，适合多用户或服务端使用。
- SGLang 部署：适合结构化生成、复杂推理流程和服务化推理。
- Ollama 部署：适合本地桌面、命令行、轻量服务和 GGUF 模型。
- llama.cpp 部署：适合 CPU、Mac、低资源环境和 GGUF 验证。
- Unsloth 推理与导出：适合从训练直接衔接到 adapter 保存、合并、GGUF 导出。

### 2.4 训练方法常用程度排名

这个排名按“新人上手概率、资料丰富度、实际项目使用频率、硬件门槛”综合排序。

| 排名 | 方法 | 常用程度 | 为什么排这里 | 先学建议 |
| --- | --- | --- | --- | --- |
| 1 | SFT | 最高 | 几乎所有模型定制都会先从 SFT 开始，数据格式直观，训练目标清楚 | 必须先学 |
| 2 | LoRA | 最高 | 大多数个人和中小团队不会做全量微调，LoRA 是最常见 adapter 路线 | SFT 后立刻学 |
| 3 | QLoRA | 很高 | 单卡显存有限时非常常用，7B、14B、20B 级模型都经常用它启动训练 | 有显存压力时优先学 |
| 4 | DPO | 高 | 偏好数据越来越常见，不需要单独奖励模型，比 PPO 容易理解 | 学完 SFT/LoRA 后学 |
| 5 | GRPO | 中高 | 推理模型、数学、代码、可验证任务中很重要，比 PPO 少 value model | 有奖励函数任务时学 |
| 6 | PPO | 中 | 经典 RLHF 方法，但组件多、成本高，新人直接上手难 | 理解 DPO/GRPO 后再学 |

解释：

- 最常用的起点是 `SFT + LoRA`。
- 显存不够时，把 LoRA 换成 QLoRA。
- 有 chosen/rejected 偏好数据时，再上 DPO。
- 有可验证奖励时，再考虑 GRPO。
- PPO 重要，但不是本仓库的第一优先实践路线，因为它对奖励模型、价值模型、KL 控制、训练稳定性要求更高。

### 2.5 量化方法常用程度排名

这个排名按“当前开源模型落地中出现频率、工具支持、部署便利性、新人理解成本”综合排序。

| 排名 | 方法 | 常用程度 | 主要用途 | 为什么排这里 | 什么时候不优先 |
| --- | --- | --- | --- | --- | --- |
| 1 | bitsandbytes 4bit | 最高 | QLoRA 训练、低显存加载 | QLoRA 路线最常见，Transformers/PEFT/TRL 生态支持成熟 | 不适合作为最终发布格式 |
| 2 | GGUF | 最高 | Ollama、llama.cpp、LM Studio 本地推理 | 本地运行最常见格式，Mac、CPU、桌面工具支持好 | 不适合作为继续训练格式 |
| 3 | bitsandbytes 8bit | 很高 | 低显存加载、简单推理验证 | 比 4bit 更接近原精度，使用简单 | 显存极紧张时不如 4bit |
| 4 | AWQ | 高 | vLLM/Transformers 推理部署 | 服务端 4bit 推理常见，速度和质量平衡好 | 需要校准数据，不适合新人第一步 |
| 5 | GPTQ | 高 | GPU 推理量化 | 老牌 4bit 推理量化格式，很多模型仓库提供 GPTQ 版本 | 新项目里常和 AWQ、GGUF 分流 |
| 6 | FP8 | 中高 | 新 GPU 服务端推理 | H100/H200/B200 等硬件上很重要，vLLM 场景常见 | 老 GPU 不一定支持 |
| 7 | EXL2 | 中 | ExLlamaV2 单机推理 | 本地 GPU 高效推理很强，但生态更集中 | 不适合 vLLM/Ollama 主线 |
| 8 | MXFP4 | 模型特定高 | gpt-oss 推理 | 对 gpt-oss 非常重要，但不是所有模型通用量化格式 | Qwen2.5 等普通模型不按这个作为主线 |

最常用结论：

- 训练最常见：`bitsandbytes 4bit`，因为它直接对应 QLoRA。
- 本地部署最常见：`GGUF`，因为 Ollama、llama.cpp、LM Studio 都围绕它。
- 服务端部署常见：`AWQ`、`GPTQ`、`FP8`，具体取决于 vLLM 支持和 GPU。
- gpt-oss 特殊重点：`MXFP4`，因为它是该模型权重压缩路线的一部分。

必须讲清楚的区别：

- bitsandbytes 4bit/8bit 是运行时加载和训练生态常用方式。
- GGUF 是本地推理文件格式。
- GPTQ/AWQ/EXL2 是离线量化后的推理模型格式。
- FP8 是服务端新硬件推理常见低精度路线。
- MXFP4 是 gpt-oss 相关的模型特定重点，不应和普通 QLoRA 4bit 混为一谈。

### 2.6 部署方式常用程度排名

这个排名按“使用频率、落地难度、适用范围、是否适合新人先跑通”综合排序。

| 排名 | 部署方式 | 常用程度 | 主要用途 | 为什么排这里 | 先学建议 |
| --- | --- | --- | --- | --- | --- |
| 1 | Transformers | 最高 | 本地验证、最小推理 | 最直接，最适合理解 tokenizer、chat template、adapter 是否生效 | 必须先学 |
| 2 | Ollama | 最高 | 本地桌面运行、快速体验 | 安装和运行简单，适合 GGUF 和官方模型 | 本地使用优先 |
| 3 | vLLM | 很高 | 服务端 API、高吞吐 | OpenAI-compatible，适合实际服务部署 | 服务化必须学 |
| 4 | llama.cpp | 高 | CPU、Mac、GGUF 验证 | GGUF 基础生态，低资源场景很常见 | 学 Ollama 时一起理解 |
| 5 | SGLang | 中高 | 结构化推理、复杂生成控制 | 适合更复杂的推理编排，但新人理解成本高于 vLLM | 掌握 vLLM 后学 |
| 6 | Unsloth 推理/导出 | 中高 | 训练后保存、合并、GGUF 导出 | 和训练链路结合紧密，适合把 adapter 变成可部署产物 | 训练路线中学习 |

最常用结论：

- 最小验证：Transformers。
- 本地直接跑：Ollama。
- 服务端部署：vLLM。
- GGUF 底层验证：llama.cpp。
- 复杂推理流程：SGLang。
- 训练到部署衔接：Unsloth。

## 3. 文件拆分原则

文件拆分必须遵守两个原则：

1. 同类内容聚合到一个文件里。
2. Qwen2.5-7B 和 gpt-oss-20b 必须分成两套独立文件。

也就是说，不要把 SFT、LoRA、QLoRA、DPO、PPO、GRPO 拆成六个很碎的 notebook。它们都属于“训练方法”，应该放在同一个训练 notebook 里，用章节区分。量化也是同理，GGUF、GPTQ、AWQ、EXL2、bitsandbytes、FP8、MXFP4 都属于“量化路线”，应该放在同一个量化 notebook 里。部署也是同理，Transformers、vLLM、SGLang、Ollama、llama.cpp 都属于“部署路线”，应该放在同一个部署 notebook 里。

但不同模型不能混写。Qwen2.5-7B 的训练、量化、部署是一套；gpt-oss-20b 的训练、量化、部署是另一套。这样新人阅读时不会在两个模型的模板、参数、硬件要求之间来回跳。

推荐每个模型固定拆成 5 个核心文件：

- `00_model_overview.ipynb`：介绍模型定位、内部结构、适用场景、限制。
- `01_training_all.ipynb`：集中讲 SFT、LoRA、QLoRA、DPO、PPO、GRPO。
- `02_quantization_all.ipynb`：集中讲 bitsandbytes、GGUF、GPTQ、AWQ、EXL2、FP8、MXFP4。
- `03_deployment_all.ipynb`：集中讲 Transformers、vLLM、SGLang、Ollama、llama.cpp。
- `04_unsloth_all.ipynb`：集中讲 Unsloth 训练、保存、合并、导出。

每个文件内部都要按“目的、输入、输出、关键参数、适合场景、常见错误”组织，不能只放代码。

文件形态优先级：

1. 优先使用 `.ipynb`。能讲解、能放参数表、能放真实代码、能按单元格执行的内容，都放到 notebook。
2. `.md` 只用于总规划、入口说明、索引和无法运行的长说明。
3. `.py` 只保留可复用脚本，不承担主要讲解任务。
4. 量化、部署、Unsloth、adapter 合并、GGUF 导出等内容，默认也放 notebook，不单独散成多个 Markdown 文件。

## 4. 内容深度标准

后续所有 Markdown、notebook、配置说明都必须按“解释清楚”来写，而不是按“能跑代码”来写。能跑只是最低要求，真正目标是让读者知道自己在做什么。

每一个训练方法、量化方法、部署方式都必须包含以下内容：

### 4.1 它是什么

必须先用中文说明概念本身，不能直接进入命令或代码。

示例要求：

- 讲 SFT 时，要说明它是用标准答案训练模型继续预测目标回答。
- 讲 LoRA 时，要说明它不是训练全部参数，而是在部分线性层旁边增加低秩矩阵。
- 讲 QLoRA 时，要说明它是“4bit 加载基座模型 + 训练 LoRA adapter”，不是把 adapter 量化成 4bit。
- 讲 AWQ 时，要说明它为什么叫激活感知量化，校准数据为什么重要。
- 讲 vLLM 时，要说明它为什么适合服务化，而不是只写 `vllm serve`。

### 4.2 为什么需要它

必须说明这个方法解决什么痛点。

示例要求：

- SFT 解决“模型不会按你的数据风格回答”的问题。
- LoRA 解决“全量训练太贵、显存不够、保存产物太大”的问题。
- QLoRA 解决“连 16bit 基座模型都很难加载”的问题。
- DPO 解决“有偏好数据但不想训练奖励模型”的问题。
- GRPO 解决“想用奖励优化，但 PPO 组件太多、显存太重”的问题。
- GGUF 解决“想在 Ollama、llama.cpp、Mac、CPU 环境里运行”的问题。
- vLLM 解决“多人请求、高吞吐、OpenAI-compatible API”的问题。

### 4.3 输入是什么

必须写清楚它需要哪些东西才能开始。

训练类输入：

- 基座模型。
- tokenizer。
- 数据集。
- chat template。
- 训练配置。
- GPU/显存要求。

量化类输入：

- 原始模型或合并后的模型。
- 校准数据。
- 量化位数。
- 输出格式。
- 目标推理后端。

部署类输入：

- 模型路径或模型名。
- adapter 路径，如果支持 adapter。
- 量化格式。
- 服务端口。
- 上下文长度。
- 推理精度。

### 4.4 输出是什么

必须写清楚做完会得到什么文件或服务。

训练类输出：

- checkpoint。
- LoRA adapter。
- tokenizer 文件。
- training logs。
- merged model，可选。

量化类输出：

- GGUF 文件。
- AWQ 模型目录。
- GPTQ 模型目录。
- EXL2 模型目录。
- FP8 或框架特定量化模型。

部署类输出：

- 本地生成结果。
- OpenAI-compatible API endpoint。
- Ollama 模型名。
- llama.cpp 可运行 GGUF。
- 可接入应用的服务地址。

### 4.5 样例数据和代码真实性

样例数据和代码必须真实可用。

样例数据要求：

- JSONL 不需要很多，给 3 到 5 条真实可用例子即可。
- 每条 JSONL 必须符合后续训练库能读取的格式。
- 不写“这里填你的数据”这种占位样本。
- 不写无法训练的半截字段。
- SFT、DPO、RL 三类数据都要分别给样例。
- 样例内容要能体现任务意图，例如格式回答、偏好对比、可验证奖励。

代码要求：

- 可以短，但必须是真的。
- 不能写伪代码。
- 不能写不存在的函数。
- 不能写“省略若干步骤”然后让读者猜。
- 不能把关键逻辑藏在注释里。
- 可以不自动运行下载大模型，但代码结构必须是实际可执行的。
- 命令示例必须使用真实 CLI 参数，不能编造参数名。

Notebook 中可以把重型步骤标成“按需执行”，但单元格里的代码仍然必须是真实代码。比如加载 `Qwen/Qwen2.5-7B-Instruct`、创建 `LoraConfig`、初始化 `SFTTrainer`、调用 `vllm serve`、调用 `ollama run`，都要使用真实库和真实参数。

### 4.6 关键参数怎么影响结果

每个参数不能只写“设置某某值”，必须解释调大、调小会发生什么。

示例要求：

- `max_seq_length`：调大能看更长上下文，但训练显存和 KV cache 占用会上升。
- `learning_rate`：太大容易训坏，太小收敛慢；LoRA 通常比全量训练可用更高学习率。
- `lora_r`：越大 adapter 表达能力越强，但参数量和显存增加。
- `temperature`：越高输出越发散，越低越稳定但可能死板。
- `top_p`：越低候选范围越窄，输出更保守。
- `gpu_memory_utilization`：vLLM 中越高越能利用显存，但太高可能 OOM。
- `num_ctx`：Ollama 中越大能保留更多上下文，但内存占用更高。

### 4.7 适合什么场景

每个方法都必须写适用场景和不适用场景。

示例要求：

- SFT 适合学习稳定格式，不适合单纯补充大量事实后期检索。
- LoRA 适合低成本行为调整，不适合极大规模知识重塑。
- QLoRA 适合低显存训练，但速度可能慢于 bf16 LoRA。
- DPO 适合偏好样本，不适合没有 rejected 数据的场景。
- PPO 适合有奖励模型或奖励函数的场景，不适合一开始就上手。
- GRPO 适合答案可验证的任务，不适合奖励很主观的开放式写作任务。
- GGUF 适合本地推理，不适合作为继续训练的标准格式。
- AWQ/GPTQ 适合推理部署，不是训练起点。

### 4.8 常用程度排名

凡是列出一组技术，都必须给出“常用程度排名”和“推荐学习顺序”。不能只把名字列出来。

排名必须说明三件事：

- 为什么常用：生态、文档、兼容性、稳定性、硬件门槛。
- 适合谁先学：新人、单卡用户、服务端部署、Mac 本地用户。
- 什么情况下不推荐：硬件不支持、格式不适合训练、生态不匹配。

注意：排名不是说某个技术绝对更强，而是说明在本仓库目标下，读者应该先理解哪个、后理解哪个。

### 4.9 常见误区

每个文件都必须有“常见误区”小节。

必须覆盖的误区：

- 以为 QLoRA 是把训练结果变成 4bit。
- 以为 LoRA adapter 单独就能部署，不需要基座模型。
- 以为 128K context 可以无成本开启。
- 以为 GGUF、GPTQ、AWQ、EXL2 可以随便互换。
- 以为 vLLM、Ollama、Transformers 是同一类东西。
- 以为 gpt-oss 的 Harmony 模板可以用普通 ChatML 替代。
- 以为 MoE 的总参数量等于每个 token 的实际计算参数量。

### 4.10 和下一步怎么衔接

每个章节最后必须说明产物下一步能去哪里。

示例：

- SFT 产出的 checkpoint 可以继续 DPO，也可以评估。
- LoRA adapter 可以合并，也可以被支持 LoRA 的推理框架加载。
- 合并后的模型可以导出 GGUF、AWQ、GPTQ。
- GGUF 可以进入 Ollama 或 llama.cpp。
- AWQ/GPTQ 可以进入 vLLM 或兼容后端。
- Transformers 验证通过后，可以迁移到 vLLM 或 SGLang。

## 5. 目录规划

```text
LLM-SFT/
  prd.md
  README.md

  data/
    sample_sft.jsonl
    sample_preference_dpo.jsonl
    sample_prompts_rl.jsonl
    README.md

  configs/
    qwen2_5_7b/
      sft.yaml
      lora.yaml
      qlora.yaml
      dpo.yaml
      ppo.yaml
      grpo.yaml
      quantization.yaml
      deploy.yaml

    gpt_oss_20b/
      sft.yaml
      lora.yaml
      qlora.yaml
      dpo.yaml
      ppo.yaml
      grpo.yaml
      quantization.yaml
      deploy.yaml

  scripts/
    qwen2_5_7b_sft.py
    gpt_oss_20b_sft.py
    merge_lora_adapter.py

  notebooks/
    qwen2_5_7b/
      00_env_check.ipynb
      01_model_overview.ipynb
      02_training_all.ipynb
      03_quantization_all.ipynb
      04_deployment_all.ipynb
      05_unsloth_all.ipynb

    gpt_oss_20b/
      00_env_check.ipynb
      01_model_overview.ipynb
      02_training_all.ipynb
      03_quantization_all.ipynb
      04_deployment_all.ipynb
      05_unsloth_all.ipynb

    shared/
      00_data_formats.ipynb
      01_lora_adapter_merge.ipynb
      02_gguf_export.ipynb
      03_awq_gptq_exl2_quantization.ipynb
      04_common_errors.ipynb
```

## 6. 两个模型介绍与内部结构

### 6.1 Qwen2.5-7B-Instruct

模型名称：`Qwen/Qwen2.5-7B-Instruct`

模型定位：

Qwen2.5-7B-Instruct 是 Qwen2.5 系列里的 7B 级别指令模型。它适合做通用对话、中文任务、英文任务、代码任务、数学任务、结构化输出任务，也适合作为本仓库第一套完整流程模型。原因是它体量适中，生态成熟，Transformers、TRL、PEFT、vLLM、SGLang、Ollama、Unsloth 都比较容易接入。

为什么适合先学这套：

- 参数规模适中，单卡训练 LoRA/QLoRA 更容易跑通。
- Dense 模型结构直观，比 MoE 更容易理解。
- Chat template 相对稳定，适合讲清楚对话数据如何变成训练 token。
- Qwen 系列中文能力强，适合中文材料。

内部结构：

- 模型类型：Causal Language Model，也就是自回归语言模型。
- 主体结构：Decoder-only Transformer。
- 位置编码：RoPE，用于让模型理解 token 的相对位置信息。
- 激活函数：SwiGLU，用于提升前馈网络表达能力。
- 归一化：RMSNorm，相比 LayerNorm 更轻量。
- 注意力结构：Grouped Query Attention，Q 头数量多，KV 头数量少，用于降低 KV cache 占用。
- QKV bias：注意力投影中带 bias，这是 Qwen2.5 模型卡明确提到的结构特征。
- 上下文长度：官方模型卡标注支持长上下文，实际训练和推理时要根据显存控制 `max_seq_length` 或 `max_model_len`。

新人需要理解的关键点：

- 它是 dense 模型。每次生成 token 时，模型大部分参数都会参与计算。
- 它是 instruct 模型。已经经过指令对齐，适合继续做对话类 SFT。
- 微调时不要破坏 chat template。训练数据要用模型对应的对话模板组织。
- 长上下文不是免费能力。上下文越长，KV cache 和训练显存占用越高。

训练时重点：

- SFT 用来让模型学习目标回答格式。
- LoRA 用来低成本改变模型行为。
- QLoRA 用来在显存有限时训练。
- DPO 用来根据好坏答案偏好继续优化。
- PPO/GRPO 用来根据奖励信号优化，GRPO 更适合可验证任务。

部署时重点：

- Transformers 用于最小验证。
- vLLM 用于服务化高吞吐。
- SGLang 用于结构化推理流程。
- Ollama/llama.cpp 用于 GGUF 本地运行。
- Unsloth 用于训练后保存、合并和导出。

### 6.2 gpt-oss-20b

模型名称：`openai/gpt-oss-20b`

模型定位：

gpt-oss-20b 是 OpenAI 发布的开放权重模型之一。它适合作为本仓库第二套模型，因为它和 Qwen2.5-7B 有明显差异：它是 MoE 模型，不是普通 dense 7B；它使用 Harmony 对话格式；它有 reasoning effort 机制；它涉及 MXFP4；它在 Ollama 和 vLLM 中有专门支持。

为什么必须单独成套：

- 它的对话模板和普通 ChatML 不一样。
- 它的推理参数里有 reasoning effort。
- 它的内部结构是 MoE，不能按 dense 模型完全解释。
- 它的量化重点包含 MXFP4，不等同于普通 bitsandbytes 4bit。
- 它的部署路线优先考虑官方兼容路径。

内部结构：

- 模型类型：Causal Language Model。
- 主体结构：Transformer-based Mixture-of-Experts。
- MoE 含义：模型内部有多个 expert。每次生成 token 时，不是所有 expert 都完整参与，而是由 router 选择部分 expert 参与计算。
- 总参数量：约 20B 级别。
- 激活参数量：约 3.6B 级别。也就是说，单个 token 实际激活的参数远少于总参数。
- 注意力机制：使用适合长上下文和高效推理的注意力设计。
- 上下文长度：官方和 Ollama 页面都标注 128K 级别上下文能力。
- 权重格式重点：大量 MoE 权重使用 MXFP4，以降低内存占用。

新人需要理解的关键点：

- MoE 的“20B”不等于每个 token 都按 20B dense 模型计算。
- router 会决定 token 分配到哪些 expert。
- active parameters 更接近单步计算成本，但总参数仍影响加载体积。
- MXFP4 是 gpt-oss 的重要压缩形式，不能简单理解成 QLoRA 的 4bit。
- Harmony 格式很重要。模板错误时，即使模型能运行，输出也可能不符合预期。

reasoning effort：

- `low`：更快，适合简单问答。
- `medium`：速度和质量平衡，适合默认使用。
- `high`：推理更充分，但输出更慢、token 更多。

训练时重点：

- 先保证 tokenizer、chat template、Harmony 格式正确。
- 优先使用 LoRA/QLoRA 或 Unsloth 路线，不建议新人一上来做全量微调。
- fp16 环境要谨慎，gpt-oss 官方和 Unsloth 文档都提到精度问题。
- 训练后验证时要检查 reasoning effort 设置是否生效。

部署时重点：

- Transformers 用于基础加载验证，但要确保 Harmony 模板正确。
- vLLM 是服务化优先路线之一，需使用支持 gpt-oss 的版本。
- Ollama 是本地运行优先路线之一，`gpt-oss:20b` 可直接运行。
- SGLang 需要以当前版本支持情况为准，如果支持不完整，应在文件中明确限制。
- Unsloth 适合 gpt-oss 本地运行、训练、保存和导出。

## 7. 数据文件规划

### 7.1 `data/sample_sft.jsonl`

用途：给 SFT、LoRA、QLoRA 使用。

推荐放 3 到 5 条真实可用样例。每一行是一个完整 JSON 对象，不能跨行。

```jsonl
{"messages":[{"role":"system","content":"你是一个回答准确、表达清楚的中文助手。"},{"role":"user","content":"用三句话解释 LoRA 的作用。"},{"role":"assistant","content":"LoRA 是一种参数高效微调方法。它会冻结大模型原有参数，只在指定层旁边训练少量低秩矩阵。这样可以明显降低显存占用和训练成本，同时保留较好的任务适配能力。"}]}
{"messages":[{"role":"system","content":"你是一个严谨的大模型工程助手。"},{"role":"user","content":"QLoRA 和 LoRA 的区别是什么？"},{"role":"assistant","content":"LoRA 通常在 16bit 或 bf16 加载的基座模型上训练低秩适配器。QLoRA 会先用 4bit 加载基座模型，再训练 LoRA 适配器，因此显存占用更低。需要注意的是，QLoRA 的训练产物仍然通常是 LoRA adapter，而不是一个 4bit adapter。"}]}
{"messages":[{"role":"system","content":"你负责把技术概念解释给初学者。"},{"role":"user","content":"为什么微调对话模型时要使用 chat template？"},{"role":"assistant","content":"chat template 用来把 system、user、assistant 等结构化消息转换成模型能理解的 token 序列。不同模型使用的角色标记和结束符可能不同，如果模板不匹配，模型会学到错误的对话边界。训练和推理使用同一个模板，能减少格式错乱和回答异常。"}]}
```

字段说明：

- `messages`：一轮或多轮对话。
- `role`：消息角色，常见值是 `system`、`user`、`assistant`。
- `content`：消息文本。
- `system`：可选，用于固定模型回答风格或任务边界。
- `user`：用户输入。
- `assistant`：希望模型学习的目标输出。

设计意图：SFT 本质是让模型学习“看到前面的消息后，应该生成 assistant 的内容”。因此数据必须清楚区分输入和目标输出。

真实可用要求：

- 每行都必须能被 `datasets.load_dataset("json", data_files=...)` 读取。
- 每条样本都必须有 assistant 目标回答。
- 样例回答不能是“略”“TODO”“自行补充”。
- 如果用于 Qwen2.5，后续 notebook 使用 Qwen tokenizer 的 chat template。
- 如果用于 gpt-oss，后续 notebook 必须先转换成 Harmony 兼容格式。

### 7.2 `data/sample_preference_dpo.jsonl`

用途：给 DPO 使用。

推荐放 3 到 5 条真实可用样例。每行包含同一个 prompt 下的一好一坏两个回答。

```jsonl
{"prompt":[{"role":"user","content":"解释 QLoRA 的优点。"}],"chosen":[{"role":"assistant","content":"QLoRA 的主要优点是显存占用低。它用 4bit 加载基座模型，同时只训练 LoRA adapter，因此单卡也更容易微调 7B、14B 甚至更大的模型。"}],"rejected":[{"role":"assistant","content":"QLoRA 和普通全量训练差不多，只是名字不同。"}]}
{"prompt":[{"role":"user","content":"什么情况下应该优先用 vLLM 部署模型？"}],"chosen":[{"role":"assistant","content":"当你需要把模型作为服务提供给多个调用方，并且关心吞吐、并发、流式输出和 OpenAI-compatible API 时，应该优先考虑 vLLM。"}],"rejected":[{"role":"assistant","content":"任何情况都必须用 vLLM，因为它总是最简单。"}]}
{"prompt":[{"role":"user","content":"LoRA adapter 是否可以脱离基座模型单独部署？"}],"chosen":[{"role":"assistant","content":"通常不可以。LoRA adapter 只保存增量参数，推理时需要和对应的基座模型一起加载，或者先把 adapter 合并进基座模型再部署。"}],"rejected":[{"role":"assistant","content":"可以，adapter 本身就是完整模型，直接运行即可。"}]}
```

字段说明：

- `prompt`：问题或上下文。
- `chosen`：更好的回答。
- `rejected`：更差的回答。

设计意图：DPO 不需要单独训练奖励模型，而是直接利用“好回答”和“差回答”的对比，让模型提高 chosen 的概率，降低 rejected 的概率。

真实可用要求：

- `chosen` 和 `rejected` 必须回答同一个 `prompt`。
- 差回答不能只是语气差，最好有明确事实错误、格式错误或缺失关键点。
- 好回答要体现想强化的行为，例如准确、具体、有边界。
- 样例要能直接喂给 TRL DPO 数据预处理逻辑。

### 7.3 `data/sample_prompts_rl.jsonl`

用途：给 PPO、GRPO 使用。

推荐放 3 到 5 条真实可用样例。优先使用可自动验证的任务，因为这类任务最容易讲清楚奖励函数。

```jsonl
{"prompt":"计算 12 + 30，只输出一个阿拉伯数字。","answer":"42","reward_type":"exact_match"}
{"prompt":"把下面内容转成 JSON，只包含 name 和 age 两个字段：张三，18岁。","answer":"{\"name\":\"张三\",\"age\":18}","reward_type":"json_schema"}
{"prompt":"判断下面 Python 表达式的结果，只输出 True 或 False：len([1,2,3]) == 3","answer":"True","reward_type":"exact_match"}
{"prompt":"写一个只包含小写字母的字符串，长度必须是 5。","answer":"regex:^[a-z]{5}$","reward_type":"regex_match"}
```

字段说明：

- `prompt`：模型要回答的问题。
- `answer`：可验证答案。
- `reward_type`：奖励计算方式，例如 `exact_match`、`regex_match`、`unit_test`、`json_schema`。

设计意图：PPO 和 GRPO 需要奖励信号。新人最容易理解的是可验证奖励，例如答案是否完全匹配、JSON 是否合法、代码测试是否通过。

真实可用要求：

- `exact_match` 可以直接比较模型输出和 `answer`。
- `json_schema` 要在 notebook 里提供真实 JSON 解析和字段校验函数。
- `regex_match` 要在 notebook 里提供真实正则匹配函数。
- 如果是代码题，必须提供真实可执行的单元测试，不写伪测试。

## 8. 配置参数规划

每个训练配置文件都要使用清晰中文注释，不能只堆参数。

### 8.1 通用参数

- `model_name`：基座模型名称。Qwen2.5 使用 `Qwen/Qwen2.5-7B-Instruct`，gpt-oss 使用 `openai/gpt-oss-20b`。
- `dataset_path`：训练数据路径。
- `output_dir`：训练输出目录，用于保存 checkpoint、adapter、日志。
- `max_seq_length`：单条样本最大 token 长度。越大越吃显存。
- `per_device_train_batch_size`：单张 GPU 每次处理的样本数。显存不足时应调小。
- `gradient_accumulation_steps`：梯度累积步数。用于用小 batch 模拟大 batch。
- `learning_rate`：学习率。LoRA/QLoRA 通常可比全量微调更高。
- `num_train_epochs`：完整遍历数据集的轮数。
- `max_steps`：最大训练步数。调试阶段可以用较小值快速验证流程。
- `warmup_ratio`：预热比例，让学习率逐步升高，减少训练初期不稳定。
- `logging_steps`：多少步打印一次日志。
- `save_steps`：多少步保存一次 checkpoint。
- `eval_steps`：多少步做一次验证。
- `bf16`：是否使用 bfloat16。A100、H100、RTX 30/40/50 系列通常更适合。
- `fp16`：是否使用 float16。老卡可能需要，但 gpt-oss 要特别注意溢出风险。
- `gradient_checkpointing`：是否用计算换显存。开启后更省显存，但训练更慢。
- `packing`：是否把多条短样本拼到一个序列中，提高训练效率。
- `assistant_only_loss`：是否只在 assistant 输出上计算 loss。对对话微调通常建议开启。

### 8.2 LoRA 参数

- `lora_r`：低秩矩阵的秩。越大可训练能力越强，但显存和参数量增加。
- `lora_alpha`：LoRA 缩放系数。常与 `lora_r` 配合使用。
- `lora_dropout`：LoRA 层 dropout，数据少时可降低过拟合。
- `target_modules`：注入 LoRA 的模块名。常见包括 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- `bias`：是否训练 bias。多数场景使用 `none`。
- `task_type`：任务类型，因果语言模型使用 `CAUSAL_LM`。

### 8.3 QLoRA 参数

- `load_in_4bit`：是否用 4bit 加载基座模型。
- `bnb_4bit_quant_type`：4bit 类型，常见 `nf4`。
- `bnb_4bit_compute_dtype`：计算精度，常见 `bfloat16`。
- `bnb_4bit_use_double_quant`：是否二次量化，通常可以进一步省显存。

设计意图：QLoRA 不是把训练结果变成 4bit adapter，而是用 4bit 加载基座模型，再训练 LoRA adapter。

### 8.4 DPO 参数

- `beta`：控制 chosen/rejected 偏好差异的强度。越大约束越强。
- `max_prompt_length`：prompt 最大长度。
- `max_completion_length`：回答最大长度。
- `loss_type`：DPO 损失类型，默认可使用标准 sigmoid DPO。

设计意图：DPO 适合“已经有成对偏好数据”的场景，例如同一个问题有一个好答案和一个坏答案。

### 8.5 PPO 参数

- `reward_model`：奖励模型路径。如果没有奖励模型，可以先用规则奖励函数。
- `kl_coef`：KL 惩罚系数，防止模型偏离原模型太远。
- `cliprange`：PPO 裁剪范围，控制每次策略更新幅度。
- `num_ppo_epochs`：每批样本上 PPO 更新次数。
- `response_length`：每次生成回答的最大长度。

设计意图：PPO 更复杂，因为通常需要策略模型、参考模型、奖励模型或奖励函数。内容中要优先讲清楚组件关系。

### 8.6 GRPO 参数

- `num_generations`：同一个 prompt 生成多少个候选回答。
- `reward_funcs`：奖励函数列表。
- `max_completion_length`：候选回答最大长度。
- `temperature`：采样温度。GRPO 需要多个不同候选，通常不能太低。

设计意图：GRPO 通过同组候选之间的相对表现更新模型，省去 PPO 中 value model 的成本，适合数学、代码、格式校验等可验证任务。

## 9. Notebook 内容规划

### 9.1 模型训练 notebook

每个训练 notebook 必须按以下结构编写：

1. 本文件解决什么问题。
2. 输入数据是什么。
3. 输出文件是什么。
4. 适合什么硬件。
5. 关键参数解释。
6. 最小可运行流程。
7. 如何判断训练是否正常。
8. 常见错误和处理方式。

### 9.2 `00_env_check.ipynb`

用途：确认环境是否能训练或推理。

需要检查：

- Python 版本。
- PyTorch 版本。
- CUDA 是否可用。
- GPU 名称和显存。
- `transformers` 版本。
- `trl` 版本。
- `peft` 版本。
- `bitsandbytes` 是否可用。
- `flash-attn` 是否可用。
- `vllm` 是否可用。
- `sglang` 是否可用。
- `ollama` 是否可用。

关键说明：

- 环境检查不是可有可无。大模型项目里大量问题来自版本、CUDA、显存和量化库不匹配。
- gpt-oss-20b 对模板和精度更敏感，环境检查中要单独标记。

### 9.3 `02_training_all.ipynb` 中的 SFT 章节

用途：用标准问答或对话数据训练模型。

适合场景：

- 让模型学习固定回答格式。
- 让模型学习某个领域的表达方式。
- 让模型更稳定地遵守指令。

输出：

- checkpoint。
- tokenizer。
- 训练日志。
- 可选 adapter。

重点参数：

- `max_seq_length`：决定单条样本最多保留多少上下文。
- `assistant_only_loss`：对话任务中只训练 assistant 部分，避免模型学习 user 内容。
- `packing`：短样本多时开启，提高训练效率。

### 9.4 `02_training_all.ipynb` 中的 LoRA 章节

用途：用较小显存完成参数高效微调。

适合场景：

- 数据量中小。
- 不想改动完整基座模型。
- 希望保存轻量 adapter。

输出：

- LoRA adapter。
- adapter config。
- 训练日志。

重点参数：

- `lora_r`：控制 adapter 表达能力。
- `target_modules`：决定 LoRA 插到哪些层。
- `lora_dropout`：控制过拟合。

### 9.5 `02_training_all.ipynb` 中的 QLoRA 章节

用途：在更低显存下训练大模型 adapter。

适合场景：

- 单卡显存有限。
- 想训练 7B、20B 或更大模型。
- 可以接受训练速度略慢。

输出：

- LoRA adapter。
- 4bit 加载配置说明。
- 训练日志。

重点参数：

- `load_in_4bit`：开启后基座模型用 4bit 加载。
- `bnb_4bit_quant_type`：推荐从 `nf4` 开始。
- `bnb_4bit_compute_dtype`：优先根据 GPU 选择 bf16 或 fp16。

### 9.6 `02_training_all.ipynb` 中的 DPO 章节

用途：用偏好数据优化回答质量。

适合场景：

- 已经有 chosen/rejected 数据。
- 想让模型减少某类差回答。
- 不想先训练奖励模型。

输出：

- DPO adapter 或 checkpoint。
- 偏好训练日志。

重点参数：

- `beta`：偏好约束强度。
- `max_prompt_length`：问题部分最大长度。
- `max_completion_length`：回答部分最大长度。

### 9.7 `02_training_all.ipynb` 中的 PPO 章节

用途：通过奖励信号进一步优化模型行为。

适合场景：

- 有奖励模型。
- 或可以写出规则奖励函数。
- 需要通过采样、评分、更新循环优化模型。

输出：

- PPO 后的 adapter 或 checkpoint。
- reward 曲线。
- KL 曲线。

重点参数：

- `kl_coef`：防止模型跑偏。
- `cliprange`：限制单次更新幅度。
- `response_length`：控制生成回答长度。

### 9.8 `02_training_all.ipynb` 中的 GRPO 章节

用途：用多个候选回答的相对奖励优化模型。

适合场景：

- 数学题。
- 代码题。
- JSON 格式任务。
- 任何可自动验证的任务。

输出：

- GRPO adapter 或 checkpoint。
- 每组候选回答奖励。
- 成功率曲线。

重点参数：

- `num_generations`：每个 prompt 生成几个候选。
- `reward_funcs`：奖励函数。
- `temperature`：控制候选多样性。

## 10. Qwen2.5-7B 特别说明

模型：`Qwen/Qwen2.5-7B-Instruct`

关键事实：

- Apache-2.0 许可。
- 参数量约 7.61B。
- 支持长上下文，模型卡标注最高可到 131K 级别。
- `transformers` 版本过低会无法识别 `qwen2`。
- 对话模板使用 Qwen 系列 ChatML 风格。

内容重点：

- 先用 Qwen2.5-7B 跑通 SFT、LoRA、QLoRA。
- 再扩展到 DPO、PPO、GRPO。
- 部署中同时覆盖 Transformers、vLLM、SGLang、Ollama。

## 11. gpt-oss-20b 特别说明

模型：`openai/gpt-oss-20b`

关键事实：

- Apache-2.0 许可。
- MoE 架构，总参数约 20B 级别，active parameters 约 3.6B。
- 官方模型卡显示支持 vLLM、Ollama、Transformers 等路线。
- Ollama 中 `gpt-oss:20b` 约 14GB，128K context。
- gpt-oss 使用 Harmony 格式，不能简单套普通 chat template。
- gpt-oss 支持 reasoning effort，可设置 low、medium、high。
- gpt-oss 权重中涉及 MXFP4，老 GPU 和 fp16 环境可能出现精度问题。

内容重点：

- 单独写 `07_harmony_and_reasoning.ipynb`。
- 明确说明 system/developer/user/assistant 结构。
- 明确说明 reasoning effort 对速度、质量、输出长度的影响。
- 训练和推理都要强调模板正确性。

## 12. 量化内容规划

### 12.1 bitsandbytes 8bit

用途：减少加载显存，常用于推理或训练前准备。

适合场景：

- 显存不够加载 bf16/fp16。
- 想快速验证模型。

关键参数：

- `load_in_8bit`：是否 8bit 加载模型。
- `device_map`：模型分配到哪些设备。

注意：

- 8bit 加载不等于生成 8bit 模型文件。
- 它更多是运行时加载方式。

### 12.2 bitsandbytes 4bit

用途：QLoRA 的核心加载方式。

适合场景：

- 低显存训练。
- 训练 LoRA adapter。

关键参数：

- `load_in_4bit`：是否 4bit 加载。
- `bnb_4bit_quant_type`：常用 `nf4`。
- `bnb_4bit_compute_dtype`：计算精度。
- `bnb_4bit_use_double_quant`：是否二次量化。

### 12.3 GGUF

用途：给 llama.cpp、Ollama、LM Studio 等本地推理工具使用。

适合场景：

- Mac 本地运行。
- CPU 推理。
- Ollama 部署。
- 轻量离线使用。

常见量化级别：

- `Q4_K_M`：常用平衡选择，体积小，质量尚可。
- `Q5_K_M`：质量更好，体积更大。
- `Q8_0`：更接近原始质量，但占用更高。

注意：

- 用户提到的 `qkkm` 通常应理解为 GGUF 里的 `Q4_K_M` 或相近 K-quant 命名。文件里要统一写标准名称，避免新人混淆。
- 如果训练得到的是 LoRA adapter，通常要先合并 adapter，再导出 GGUF。

### 12.4 GPTQ

用途：离线权重量化，主要用于推理。

适合场景：

- GPU 推理。
- 需要比 fp16 更省显存。
- 使用 AutoGPTQ 或兼容运行时。

关键参数：

- `bits`：量化位数，常见 4bit。
- `group_size`：分组大小，常见 128。
- `desc_act`：是否使用 activation order，可能影响质量和速度。

注意：

- GPTQ 通常不用于继续训练。
- 更适合模型定型后的推理发布。

### 12.5 AWQ

用途：激活感知权重量化，常用于高效推理。

适合场景：

- vLLM 推理。
- 服务端压缩部署。
- 希望在速度和质量之间取得平衡。

关键参数：

- `w_bit`：权重量化位数，常见 4。
- `q_group_size`：量化分组大小，常见 128。
- `zero_point`：是否使用 zero point。
- `version`：量化实现版本，例如 GEMM。

注意：

- AWQ 需要校准数据。
- 校准数据应接近真实输入分布。

### 12.6 EXL2

用途：ExLlamaV2 生态的高效推理格式。

适合场景：

- 单机 GPU 推理。
- 追求本地推理速度。
- 使用 text-generation-webui 或 ExLlamaV2 相关工具。

关键参数：

- `bpw`：bits per weight，表示平均每个权重使用多少 bit。
- `calibration dataset`：校准数据，影响量化质量。

注意：

- EXL2 生态和 vLLM/Ollama 不是同一条路线。
- 内容里要讲清楚它适合本地推理，不是训练格式。

### 12.7 FP8

用途：服务端高吞吐推理中的低精度方案。

适合场景：

- H100、H200、B200 等新硬件。
- vLLM 服务部署。
- 多用户推理。

关键参数：

- `quantization`：vLLM 中指定量化方式。
- `kv_cache_dtype`：KV cache 精度。
- `tensor_parallel_size`：张量并行数量。

注意：

- FP8 依赖硬件和推理框架支持。
- 不是所有 GPU 都适合。

### 12.8 MXFP4

用途：gpt-oss 系列的重要压缩格式。

适合场景：

- gpt-oss-20b 本地推理。
- Ollama 运行 gpt-oss。
- 支持 MXFP4 的推理后端。

注意：

- MXFP4 和普通 bitsandbytes 4bit 不是一回事。
- gpt-oss 的 MoE 权重使用该格式降低内存占用。
- 老 GPU 上可能需要框架内部做精度转换，速度和稳定性会受影响。

## 13. 部署内容规划

### 13.1 Transformers 部署

用途：最基础的本地加载和生成方式。

适合场景：

- 验证模型能否加载。
- 验证训练后的 adapter 是否生效。
- 检查 tokenizer 和 chat template 是否正确。
- 小规模单用户推理。

核心参数：

- `model_name`：模型或本地 checkpoint 路径。
- `torch_dtype`：模型加载精度，例如 `auto`、`bfloat16`、`float16`。
- `device_map`：模型放到哪些设备，常用 `auto`。
- `max_new_tokens`：最多生成多少新 token。
- `temperature`：采样随机性，越高越发散。
- `top_p`： nucleus sampling 参数，控制候选 token 累积概率。
- `do_sample`：是否采样。关闭时更接近贪心输出。

Qwen 注意点：

- 使用 `tokenizer.apply_chat_template`。
- 需要确保 EOS token 和 chat template 匹配。

gpt-oss 注意点：

- 必须使用正确 Harmony 模板。
- reasoning effort 应在示例中明确 low、medium、high 的区别。

### 13.2 vLLM 部署

用途：把模型部署成 OpenAI-compatible API 服务。

适合场景：

- 多用户访问。
- 高吞吐服务。
- 后端系统调用。
- 需要 streaming、batching、KV cache 管理。

核心参数：

- `model`：模型名称或本地路径。
- `host`：服务监听地址，例如 `0.0.0.0`。
- `port`：服务端口，例如 `8000`。
- `dtype`：推理精度，例如 `auto`、`bfloat16`、`float16`。
- `max_model_len`：最大上下文长度。越大 KV cache 占用越高。
- `gpu_memory_utilization`：允许 vLLM 使用的 GPU 显存比例。
- `tensor_parallel_size`：张量并行卡数。
- `quantization`：量化方式，例如 AWQ、GPTQ、FP8 等。
- `enable_lora`：是否启用 LoRA adapter 加载。
- `served_model_name`：API 暴露给调用方的模型名。

Qwen 示例意图：

- 展示如何用 vLLM 加载 `Qwen/Qwen2.5-7B-Instruct`。
- 展示如何用 OpenAI SDK 或 curl 调用 `/v1/chat/completions`。

gpt-oss 示例意图：

- 使用官方推荐的 gpt-oss 兼容 vLLM 版本。
- 说明 Harmony 和 reasoning effort 对服务输出的影响。

### 13.3 SGLang 部署

用途：部署结构化推理和复杂生成流程。

适合场景：

- 多轮推理流程。
- 结构化输出。
- 需要更强控制生成过程。
- 服务端推理。

核心参数：

- `model-path`：模型名称或本地路径。
- `host`：监听地址。
- `port`：监听端口。
- `tp`：tensor parallel 数量。
- `mem-fraction-static`：静态显存占用比例。
- `context-length`：最大上下文长度。
- `dtype`：推理精度。
- `quantization`：量化方式，如果后端支持。

Qwen 示例意图：

- 展示 SGLang 启动 Qwen2.5-7B 的服务。
- 展示 OpenAI-compatible API 调用方式。

gpt-oss 示例意图：

- 先确认 SGLang 当前版本对 gpt-oss 的支持程度。
- 如果支持，给出启动和调用说明。
- 如果支持不完整，明确写出限制，并建议使用 vLLM 或 Ollama 作为优先路线。

### 13.4 Ollama 部署

用途：本地一条命令运行模型。

适合场景：

- 桌面本地使用。
- 快速体验。
- 和 Open WebUI、桌面工具集成。
- 运行 GGUF 或 Ollama 官方模型。

核心参数：

- `model`：Ollama 模型名，例如 `gpt-oss:20b`。
- `num_ctx`：上下文长度。
- `temperature`：采样随机性。
- `top_p`：采样范围。
- `repeat_penalty`：重复惩罚。
- `num_gpu`：使用多少 GPU 层或 GPU 资源，具体取决于平台。

Qwen 注意点：

- 如果使用官方或社区 Qwen GGUF，需要说明模型来源。
- 如果使用自己训练的 adapter，需要先 merge，再转 GGUF，再创建 Ollama Modelfile。

gpt-oss 注意点：

- 官方 Ollama 已支持 `gpt-oss:20b`。
- `ollama run gpt-oss:20b` 是最简单路线。
- reasoning effort 和工具能力要按 Ollama 当前支持能力说明。

### 13.5 llama.cpp 部署

用途：运行 GGUF 模型。

适合场景：

- CPU 推理。
- Mac Metal 推理。
- 本地轻量部署。
- 验证 GGUF 导出是否成功。

核心参数：

- `model`：GGUF 文件路径。
- `ctx-size`：上下文长度。
- `n-gpu-layers`：放到 GPU 的层数。
- `threads`：CPU 线程数。
- `temp`：采样温度。
- `top-p`：采样范围。

注意：

- llama.cpp 是 GGUF 路线的基础工具。
- Ollama 很多本地模型工作流也依赖 GGUF 思路。

### 13.6 Unsloth 部署与导出

用途：把训练结果保存、合并、导出到推理工具。

适合场景：

- 用 Unsloth 训练 LoRA/QLoRA。
- 需要保存 adapter。
- 需要合并 adapter 到基座模型。
- 需要导出 GGUF 给 Ollama 或 llama.cpp。

核心参数：

- `load_in_4bit`：是否 4bit 加载。
- `max_seq_length`：训练或推理上下文长度。
- `FastLanguageModel.for_inference`：切换到推理模式。
- `save_pretrained`：保存 adapter 或模型。
- `save_pretrained_merged`：保存合并后的模型。
- `save_pretrained_gguf`：导出 GGUF。

注意：

- adapter 很小，但部署时通常需要基座模型配合。
- 如果要给 Ollama 使用，通常要合并并导出 GGUF。

## 14. 脚本与 Notebook 关系规划

本项目优先使用 notebook 作为主要载体。脚本只承担“可复用命令行工具”的角色，不承担主要讲解角色。

原则：

- 讲解放在 notebook。
- 参数解释放在 notebook。
- 最小可运行流程放在 notebook。
- 可复用批处理逻辑再抽成 `.py`。
- 不再把 GGUF、AWQ、GPTQ、EXL2 这类内容拆成独立 `.md`，统一放到 `notebooks/shared/` 或对应模型的量化 notebook。

### 14.1 `scripts/qwen2_5_7b_sft.py`

用途：提供一个可命令行运行的 Qwen2.5-7B SFT 脚本。

对应 notebook：

- `notebooks/qwen2_5_7b/02_training_all.ipynb`

关系说明：

- notebook 负责解释 SFT、LoRA、QLoRA、DPO、PPO、GRPO。
- 脚本只保留 SFT 的命令行版本，方便后续不打开 notebook 也能跑一次标准训练。

参数说明：

- `--model_name`：模型名，默认 `Qwen/Qwen2.5-7B-Instruct`。
- `--data_path`：SFT JSONL 数据路径。
- `--output_dir`：输出目录。
- `--max_seq_length`：最大长度。
- `--learning_rate`：学习率。
- `--num_train_epochs`：训练轮数。
- `--packing`：是否启用 packing。
- `--assistant_only_loss`：是否只训练 assistant 内容。

### 14.2 `scripts/gpt_oss_20b_sft.py`

用途：提供一个可命令行运行的 gpt-oss-20b SFT 脚本。

对应 notebook：

- `notebooks/gpt_oss_20b/02_training_all.ipynb`

额外说明：

- 必须处理 Harmony 模板。
- 必须清楚标注 reasoning effort。
- 必须说明 bf16、fp16、MXFP4 相关限制。

### 14.3 `scripts/merge_lora_adapter.py`

用途：把 LoRA adapter 合并到基座模型。

对应 notebook：

- `notebooks/shared/01_lora_adapter_merge.ipynb`

输入：

- 基座模型路径。
- adapter 路径。
- 输出路径。

输出：

- 合并后的 Hugging Face 格式模型。

设计意图：很多部署工具不直接加载 adapter，或者加载 adapter 的方式不同。合并模型能减少部署差异。

### 14.4 `notebooks/shared/02_gguf_export.ipynb`

用途：说明并演示如何把合并后的模型导出为 GGUF。

必须解释：

- 为什么要先合并 adapter。
- GGUF 和 Hugging Face safetensors 的区别。
- `Q4_K_M`、`Q5_K_M`、`Q8_0` 的选择。
- 如何验证 GGUF 能运行。

### 14.5 `notebooks/shared/03_awq_gptq_exl2_quantization.ipynb`

用途：集中说明 AWQ、GPTQ、EXL2 量化流程。

AWQ 必须解释：

- 校准数据是什么。
- 为什么校准数据会影响量化质量。
- AWQ 更适合推理而不是训练。
- vLLM 中如何加载 AWQ。

GPTQ 必须解释：

- `bits`、`group_size`、`desc_act` 含义。
- GPTQ 和 AWQ 的区别。
- 适合哪些推理后端。

EXL2 必须解释：

- `bpw` 是什么。
- 为什么 EXL2 适合 ExLlamaV2 生态。
- 它和 vLLM/Ollama 路线的区别。

## 15. README 规划

`README.md` 后续需要作为入口索引。

必须包含：

- 项目能做什么。
- 推荐先跑哪几个文件。
- 硬件最低建议。
- 数据格式说明。
- Qwen2.5-7B 路线。
- gpt-oss-20b 路线。
- 训练路线选择表。
- 量化路线选择表。
- 部署路线选择表。
- 常见问题。

## 16. 路线选择表

### 16.1 训练路线

| 目标 | 推荐路线 | 原因 |
| --- | --- | --- |
| 第一次跑通 | Qwen2.5-7B + SFT | 模型结构常见，资料多，问题少 |
| 显存有限 | QLoRA | 4bit 加载基座模型，显存压力低 |
| 只想保存小文件 | LoRA | adapter 小，便于保存和分享 |
| 有好坏答案对 | DPO | 不需要单独奖励模型 |
| 有奖励模型 | PPO | 可以利用奖励模型优化输出 |
| 有可验证答案 | GRPO | 省去 value model，适合数学、代码、格式任务 |
| 训练 gpt-oss | Unsloth 或官方兼容 TRL 路线 | 模板和精度细节更多，需要更稳的封装 |

### 16.2 量化路线

| 目标 | 推荐路线 | 原因 |
| --- | --- | --- |
| QLoRA 训练 | bitsandbytes 4bit | 训练生态成熟 |
| Transformers 低显存加载 | bitsandbytes 8bit/4bit | 改动少，容易验证 |
| Ollama 本地运行 | GGUF | Ollama/llama.cpp 常用 |
| vLLM 服务端推理 | AWQ、GPTQ、FP8 | vLLM 支持多种推理量化 |
| 单机高效推理 | EXL2 | ExLlamaV2 生态速度好 |
| gpt-oss 本地运行 | MXFP4 或官方 Ollama 模型 | gpt-oss 原生重点格式 |

### 16.3 部署路线

| 目标 | 推荐路线 | 原因 |
| --- | --- | --- |
| 验证模型可用 | Transformers | 最直接，便于调试 |
| API 服务 | vLLM | OpenAI-compatible，高吞吐 |
| 复杂生成控制 | SGLang | 适合结构化推理流程 |
| 本地桌面使用 | Ollama | 安装和运行简单 |
| CPU/Mac/低资源 | llama.cpp | GGUF 支持好 |
| 从训练到导出 | Unsloth | 训练、保存、合并、导出衔接顺 |

## 17. 验收标准

后续内容完成时，需要满足：

- 每个 notebook 都能独立说明目的、输入、输出、关键参数。
- 每个配置文件都有中文注释。
- 每个训练方法都有 Qwen2.5-7B 和 gpt-oss-20b 两套内容。
- 每个部署方式都包含 Qwen2.5-7B 和 gpt-oss-20b 的差异说明。
- 量化章节清楚区分训练时量化、推理时量化、文件格式量化。
- 新人可以先跑 `00_env_check.ipynb`，再按 README 路线选择继续执行。
- gpt-oss 相关内容必须明确 Harmony、reasoning effort、MXFP4、精度限制。
- Qwen 相关内容必须明确 chat template、EOS、Transformers 版本要求。

## 18. 参考来源

- Qwen2.5-7B-Instruct 模型卡：`https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`
- gpt-oss-20b 模型卡：`https://huggingface.co/openai/gpt-oss-20b`
- TRL 文档：`https://huggingface.co/docs/trl`
- vLLM 文档：`https://docs.vllm.ai`
- SGLang 文档：`https://docs.sglang.ai`
- Ollama gpt-oss：`https://ollama.com/library/gpt-oss`
- Unsloth 文档：`https://unsloth.ai/docs`
- llama.cpp：`https://github.com/ggml-org/llama.cpp`
