# Profiling：三个 draft 的投机解码阶段剖析（nsys）

同一套打桩、同一条输入、同一张卡，对比 EAGLE-3 / DFlash / DSpark 在**单条 prefill + decode** 下的
时间分配与 SM 利用率。

## 采集方式

```bash
export NSYS_TMPDIR=/mnt/infini/zym-tmp/nsys_tmp      # /tmp 只剩 91MB，必须换
cd DeepSpec-lab && . .venv/bin/activate
export HF_HOME=/mnt/infini/zym-tmp/hf_cache PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=2 PROF_DRAFT=<draft> PROF_MAX_NEW_TOKENS=512 PROF_OUTDIR=... \
nsys profile --trace=cuda,nvtx --gpu-metrics-devices=cuda-visible \
  --sample=none --cpuctxsw=none -o .../nsys_<tag> python ../profile_single.py
python ../analyze_nsys.py .../nsys_<tag>.sqlite
```

- **nsys 2026.6.1** —— apt 里 Ubuntu multiverse 的 `nsight-systems` 只有 **2022.4，没有 `--gpu-metrics-devices`**；改从 NVIDIA devtools 源取 `.deb`
- `--gpu-metrics-devices=cuda-visible`（nsys 提示的用法），**有效采样间隔 ≈ 100 µs**
- 打桩全部用 **monkeypatch，未改 DeepSpec 框架任何一行代码**（`server/profile_single.py`）
- 预热阶段单独打 `WARMUP:` 前缀，与正式测量严格隔离
- **不使用 nvidia-smi**：其 `utilization.gpu` 只是"有 kernel 在跑的时间占比"，粒度粗且跨阶段污染

## 协议

| 项 | 值 |
|---|---|
| target | `Qwen/Qwen3-4B` |
| draft | `eagle3_qwen3_4b_ttt7` / `dflash_qwen3_4b_block7` / `dspark_qwen3_4b_block7` |
| 输入 | 1 条 gsm8k prompt，**92 tokens** |
| 生成 | `max_new_tokens=512`，batch=1 |
| 机器 | GPU 2 独占（0/1 被其它 vLLM 任务占用，未使用） |

## 1. 绝对耗时（整段 decode）

| | EAGLE-3 | DFlash | DSpark |
|---|---|---|---|
| decode 次数 / 生成 token | 48 / 289 | 52 / 275 | 49 / 306 |
| `TARGET_VERIFY` | 2110.2 ms | 2146.0 ms | 1958.6 ms |
| `DRAFT_PROPOSE` | **830.6 ms** | **343.2 ms** | **412.3 ms** |
| `DRAFT_UPDATE` | 81.5 ms | 1.6 ms | 1.5 ms |
| `POST_VERIFY`（仅 DSpark，测量开销） | — | — | 23.6 ms |
| `PREFILL`（单次） | 36.0 ms | 36.6 ms | 36.4 ms |
| 循环总时长 | 3022 ms | 2491 ms | 2372 ms |
| **draft : verify** | **0.432** | **0.161** | **0.211** |

## 2. 归一化到「每次 decode」（消除迭代次数差异）

| 每 iteration | EAGLE-3 | DFlash | DSpark |
|---|---|---|---|
| `TARGET_VERIFY` | 43.96 ms | 41.27 ms | 39.97 ms |
| `DRAFT_PROPOSE` | **17.31 ms** | **6.60 ms**（**2.62×** 快） | **8.41 ms**（1.65× 快） |
| `DRAFT_UPDATE` | 1.70 ms | **0.03 ms**（51× 快） | 0.03 ms |
| **draft 小计** | 19.01 ms | 6.63 ms | 8.44 ms |
| 每 iteration 合计 | 62.97 ms | 47.90 ms | 48.41 ms |

## 3. SM 利用率（nsys GPU metrics）

`SMs Active [Throughput %]` = 平均 SM 活跃程度；`SM Issue` = 指令发射利用率；
`Tensor Active` = tensor core 利用率；`DRAM Rd/Wr` = 显存读写带宽占峰值比例。

| 阶段 | EAGLE-3 | DFlash | DSpark |
|---|---|---|---|
| **整段 decode：SM Active** | 18.9% | 16.9% | 17.3% |
| 整段 decode：Tensor Active | 1.0% | 1.3% | 1.3% |
| 整段 decode：SM Issue | 2.1% | 1.6% | 1.8% |
| 整段 decode：DRAM Rd | 15.7% | 13.3% | 13.3% |
| draft 阶段：SM Active | 27.5% | 20.0% | 20.0% |
| draft 阶段：**DRAM Rd** | **25.8%** | **16.7%** | **14.9%** |
| verify 阶段：SM Active | 15.2% | 16.4% | 16.8% |
| verify 阶段：DRAM Rd | 11.8% | 12.7% | 12.9% |
| PREFILL：Tensor Active | 6.1% | — | — |

## 4. 结论

1. **SM 利用率三者都在 ~17-19%，tensor core 仅 ~1%，指令发射仅 ~2%。**
   batch=1 的 decode 是彻底的 memory-bound —— 瓶颈不在算力，在访存与 kernel 发射。
   这是投机解码"加速比上不去"的硬件层原因，也说明**单纯优化 draft 结构改变不了大盘**。

2. **块并行对自回归 draft 的优势被量化出来了：`DRAFT_PROPOSE` 每 iteration 17.31 → 6.60 ms（2.62×）。**
   EAGLE-3 是 7 次串行的单 token 前向，单 token GEMM 是 1×4096 瘦矩阵，
   **每次都要把 draft 权重从显存整读一遍，7 次就是 7 遍** —— 证据就是 draft 阶段
   **DRAM Rd 25.8% vs DFlash 16.7%**（近乎腰斩）。
   这从硬件计数器上坐实了 DFlash 论文"draft 成本 ∝ γ"的论断。

3. **`DRAFT_UPDATE` 差 51×（1.70 ms → 0.03 ms）**，是一个容易忽略的次要来源：
   EAGLE-3 要对 committed tokens 调 `extend_draft_cache` 重算 draft hidden；
   DFlash/DSpark 只对已算好的 context feature 做一次切片。

4. **DSpark 比 DFlash 在 draft 上慢 ~27%（6.60 → 8.41 ms/iter），这是 Markov head 的代价**
   —— 它用一次轻量串行前向换更高的接受长度（见 [qwen3-4b-t1.md](qwen3-4b-t1.md)：τ 4.04 → 4.74）。
   **这是一次明码标价的交易，也是它相对 DFlash 的核心设计取舍。**

5. **归一化后三者的 verify 耗时几乎相同（39.97 / 41.27 / 43.96 ms）**，因为都是同一个
   target 在同规模输入下的前向。**verify 占 DFlash/DSpark 的 82-86%、EAGLE-3 的 70%。**
   → **加速器做得越省，杠杆越往验证端转移。** 仅优化 draft，在最省的那一端收益上限已不到 14%；
   要再往前推必须动验证（变长验证 / 跳过低置信位置）—— DSpark 的 confidence-scheduled
   verification 精确瞄准的就是这个位置。

## 5. 口径与局限（引用务必带上）

- **单条样本、batch=1**，用于看**结构与瓶颈**，不代表服务端吞吐表现。
- 三个 draft 的迭代次数不同（48/52/49），**跨模型比较请用第 2 节的每-iteration 归一化表**；
  第 1 节的绝对值受迭代次数影响。
- `SMs Active` / `SM Issue` 是 nsys GPU metrics 的**采样子**（100 µs），非精确指令计数；
  要更硬的数可用 ncu 的 `sm__throughput.avg.pct_of_peak_sustained_elapsed`（逐 kernel 计数）。
- 阶段耗时是 **CPU 侧 NVTX range 时长**（含 kernel 发射开销）；与 GPU 侧投影时间
  （`nvtx_gpu_proj_sum`）基本一致，说明发射开销不是主要偏差。
- DSpark 的 `POST_VERIFY` 是**置信度头记录器的测量开销，不属于算法本身**，已单列。
