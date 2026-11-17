# scripts/profiling

剖析投机解码的两个问题：**各阶段 SM 利用率**、**draft 与 target verify 的时间分配**。

采集用 **nsys**（不用 nvidia-smi —— 它的 `utilization.gpu` 只是"有 kernel 在跑的时间占比"，
粒度粗且跨阶段污染）。打桩全部走 **monkeypatch，不改 DeepSpec 框架任何一行代码**。

## 文件

| 文件 | 作用 |
|---|---|
| `profile_single.py` | 单条 gsm8k 输入跑 prefill + decode，给 4 个阶段压 NVTX range，并记录逐阶段起止时间 |
| `analyze_nsys.py` | 读 nsys 的 sqlite 导出，把 NVTX range 和 GPU metrics 时间线对齐，出阶段表 |
| `run_nsys.sh` | 三个 draft 依次采集的驱动脚本 |

## 打桩的四个阶段

| NVTX range | 对应代码 | 含义 |
|---|---|---|
| `PREFILL` | `generate_decoding_sample` 里 target 的第 1 次 forward | 首 token 的 prefill |
| `DRAFT_PROPOSE` | evaluator 的 `_propose` | 产候选 token（**EAGLE-3 是 7 次串行前向；DFlash/DSpark 是 1 次块前向**） |
| `TARGET_VERIFY` | `base_evaluator.verify_draft_tokens` | target 并行验证 |
| `DRAFT_UPDATE` | evaluator 的 `_update` | 验证后续接 draft 状态 |
| `POST_VERIFY` | DSpark 的 `_post_verify` | 置信度头记录 —— **属于测量开销，不是算法本身**，单独看 |

预热阶段的 range 带 `WARMUP:` 前缀，与正式测量严格隔离。

## 跑法

```bash
# 1) 装 nsys >= 2023.1（GPU metrics 那时才引入）
#    apt 里 Ubuntu multiverse 的 nsight-systems 只有 2022.4，没有 --gpu-metrics-devices。
#    从 NVIDIA devtools 源取 .deb：
#    https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/
export NSYS_TMPDIR=/path/to/big/disk/nsys_tmp   # nsys 要 >200MB 临时空间，/tmp 常常不够

# 2) 采集（仓库根目录）
bash scripts/profiling/run_nsys.sh

# 3) 或者只跑一个 draft / 单条
GPU=2 MAX_NEW_TOKENS=512 DRAFTS="deepseek-ai/eagle3_qwen3_4b_ttt7" \
  bash scripts/profiling/run_nsys.sh

# 4) 单独分析已有的采集
python scripts/profiling/analyze_nsys.py prof/eagle3_qwen3_4b_ttt7/nsys_eagle3_qwen3_4b_ttt7.sqlite
```

## 口径与限制（引用务必带上）

- 结果见 [`docs/baselines/profiling.md`](../../docs/baselines/profiling.md)。
- **单条样本、batch=1**，用于看结构与瓶颈，不代表服务端吞吐。
- `SMs Active [Throughput %]` 是 nsys GPU metrics 的**采样**量（实测有效间隔 ≈ 100 µs），
  非精确指令计数。要更硬的数用 ncu 的
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`（逐 kernel 计数）。
- 阶段耗时是 **CPU 侧 NVTX range 时长**（含 kernel 发射开销）；实测与 GPU 侧投影时间
  （`nsys stats --report nvtx_gpu_proj_sum`）基本一致。
- 同一张卡必须独占（`CUDA_VISIBLE_DEVICES` 只留一张），否则 GPU metrics 会被别的进程污染。
