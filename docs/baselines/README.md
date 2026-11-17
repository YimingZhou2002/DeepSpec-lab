# 对比基线

固化下来的评测结果，作为后续每次改动、以及接入我们自己的方法时的对照基准。

## 协议（复现必须逐项满足）

| 项 | 值 |
|---|---|
| target | `Qwen/Qwen3-4B` |
| 采样 | `temperature=1.0`、`max_new_tokens=2048`、`seed=980406` |
| 数据集 | `eval.py` 默认 9 个 task，共 3030 条 prompt |
| 指标 | `acceptance_length`(τ)、`draft_tokens_per_proposal`、`verify_rate`、`accept_rates_by_position` |
| 硬件 | 8× NVIDIA（97887 MiB/卡）· CUDA 13.0 · driver 580.159.04 |
| 软件 | torch 2.11.0+cu130 · transformers 5.10.2 |
| 代码 | `DeepSpec-lab @ 005e03b` |

```bash
cd DeepSpec-lab && . .venv/bin/activate
# 全量（9 数据集）
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python eval.py \
  --target_name_or_path Qwen/Qwen3-4B --draft_name_or_path <draft> --temperature 1.0
# 单数据集（快速迭代用）
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/eval_subset.py <dataset> <limit> \
  Qwen/Qwen3-4B <draft> 1.0
```

## 新增对比时的公平性要求

1. **同 target、同温度、同 `max_new_tokens`** —— 三者任一不同，数字不可比。
2. **标注提案预算**：本基线三个 draft 都是 block/ttt = 7（每次提案 6.9~7.0 个 token）。我们的方法若改了提案长度，必须同时报预算，否则 τ 的比较没有意义。
3. **同机对比**：换卡后绝对 τ 仍可比（τ 是算法量），但**端到端加速比不可比**，只能比相对提升。
4. 结论要报 **macro-avg τ**，不要只挑好看的子集。

## 已有基线

| 文件 | 内容 |
|---|---|
| [qwen3-4b-t1.md](qwen3-4b-t1.md) | Qwen3-4B / T=1.0 / 三模型 × 9 数据集（人读） |
| [qwen3-4b-t1.json](qwen3-4b-t1.json) | 同上，机器可读，便于 diff |
| [profiling.md](profiling.md) | 三模型单条输入的 **时间分配 + SM 利用率**（nsys 采集） |

## 与论文的一致性（可信度证据）

| 对比 | 本基线实测 | DSpark 论文 (Qwen3-4B) |
|---|---|---|
| DSpark vs Eagle3 | +31.4% | +30.9% |
| DSpark vs DFlash | +17.3% | +18.3% |

两个相对提升量与论文几乎重合 ⇒ 这套环境可以直接当可信基线用。

## 待补

- **T=0 那一档**（论文双温度报告）。T=1 已覆盖。
- 其它 target（`Qwen3-8B`、`Qwen3-14B`、`Gemma4-12B`）—— config 已存在，未跑。
