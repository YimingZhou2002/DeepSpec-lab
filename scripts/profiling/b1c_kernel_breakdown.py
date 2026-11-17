"""B1c: 那 24.5 ms 到底是 GPU 在干活，还是在等？

B1/B1b 已确定：target 一次前向固定 ~24.5 ms，与验证 token 数无关、与 batch 无关，
每次都要把 7.49 GiB 权重流一遍（有效带宽 ≈ 327 GB/s，远低于峰值）。

剩下唯一的机制问题：这 24.5 ms 里
  (a) GPU kernel 实际执行了多少时间？
  (b) 有多少是 kernel 之间的空隙（发射/依赖停顿）？

判据
----
用 nsys 取 `cuda_gpu_kern_sum` 的总 kernel 时间，与脚本自己测的 wall time 相减：
- 若 kernel 总时间 ≈ wall time  ⇒ GPU 真的在忙，是访存延迟受限（stall 在 kernel 内）
- 若 kernel 总时间 << wall time  ⇒ 大量空隙，是发射/同步开销受限

两种结论对应**完全不同的优化方向**，所以必须先测不能再猜。

用法
----
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      -o prof/b1c python scripts/profiling/b1c_kernel_breakdown.py prof/b1c_wall.json
    nsys stats --report cuda_gpu_kern_sum prof/b1c.nsys-rep
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

TARGET = os.environ.get("B1C_TARGET", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("B1C_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
L = int(os.environ.get("B1C_L", "92"))
N = int(os.environ.get("B1C_N", "8"))
ITERS = int(os.environ.get("B1C_ITERS", "50"))
PHASE = "B1C_ONESHOT"


def main(out_path: str) -> None:
    sys.argv = ["eval.py", "--target_name_or_path", TARGET,
                "--draft_name_or_path", DRAFT, "--temperature", "1.0"]
    import eval as E  # noqa: E402

    arch = AutoConfig.from_pretrained(DRAFT).architectures[0]
    args = E.parse_args()
    args.tasks = []
    evaluator = E.EVALUATORS[arch](0, args)
    model = evaluator.target_model
    cfg = model.config
    dev = model.device

    ids = torch.randint(0, cfg.vocab_size, (1, L + N + 8), device=dev)
    nid, pos = ids[:, L:L + N], torch.arange(L, L + N, device=dev)[None]

    def prefill():
        c = DynamicCache()
        with torch.no_grad():
            o = model(input_ids=ids[:, :L], past_key_values=c,
                      use_cache=True,
                      position_ids=torch.arange(L, device=dev)[None])
        return o.past_key_values

    # 预热
    for _ in range(5):
        c = prefill()
        with torch.no_grad():
            model(input_ids=nid, past_key_values=c, use_cache=True, position_ids=pos)
    torch.cuda.synchronize()

    # 计时 + 打 NVTX，只包住「一次 N-token 前向」
    ts = []
    for _ in range(ITERS):
        cache = prefill()                     # 不计时
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(PHASE)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=nid, past_key_values=cache, use_cache=True, position_ids=pos)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        torch.cuda.nvtx.range_pop()

    out = {"meta": {"target": TARGET, "ctx_len": L, "N": N, "iters": ITERS,
                    "nvtx_phase": PHASE},
           "wall_ms_median": statistics.median(ts) * 1e3,
           "wall_ms_min": min(ts) * 1e3,
           "wall_ms_sum": sum(ts) * 1e3}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b1c_wall.json")
