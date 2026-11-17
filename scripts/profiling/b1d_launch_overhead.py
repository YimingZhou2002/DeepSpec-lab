"""B1d: 空隙假说的验证 —— 减少 kernel 数量能不能把 24 ms 压下来？

B1c 的结论：一次前向有 2309 次 kernel 发射、平均每个只跑 4.5 µs、GPU 总执行
10.4 ms，而 wall 是 28.7 ms ⇒ **约 63% 的时间是 kernel 之间的空隙**。

若这个假说成立，那么「把 op 图折叠掉」（CUDA graph / 算子融合）应当带来**数倍**
而不是百分之几十的提升。这跟投机解码能给的收益完全不是一个量级，必须实测。

做法：同一个固定形状的前向，比较
  (a) eager
  (b) torch.compile(mode="reduce-overhead")  —— 内部用 CUDA graph 消除逐 kernel 发射

固定形状是前提（变长会破坏 graph），所以这里用 use_cache=False 的定长前向；
op 图与 verify 那次前向同构，足以回答「空隙占比」这个问题。

用法
----
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/b1d_launch_overhead.py out.json
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

TARGET = os.environ.get("B1D_TARGET", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("B1D_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
TOTAL_LEN = int(os.environ.get("B1D_LEN", "100"))   # 92 context + 8 个待验证 token
REPS = int(os.environ.get("B1D_REPS", "30"))


def bench(fn, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, min(ts) * 1e3


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

    ids = torch.randint(0, cfg.vocab_size, (1, TOTAL_LEN), device=dev)
    pos = torch.arange(TOTAL_LEN, device=dev)[None]

    def eager():
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, position_ids=pos)

    out = {"meta": {"target": TARGET, "total_len": TOTAL_LEN, "reps": REPS},
           "eager_ms": None, "compiled_ms": None}

    out["eager_ms"], out["eager_min_ms"] = bench(eager)
    print(f"[b1d] eager           = {out['eager_ms']:.2f} ms", flush=True)

    # CUDA graph：手工捕获，形状固定
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, position_ids=pos)

        def replay():
            g.replay()

        out["cudagraph_ms"], out["cudagraph_min_ms"] = bench(replay)
        print(f"[b1d] CUDA graph replay = {out['cudagraph_ms']:.2f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        out["cudagraph_error"] = f"{type(e).__name__}: {e}"
        print(f"[b1d] CUDA graph 失败: {out['cudagraph_error']}", flush=True)

    try:
        cm = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        def compiled():
            with torch.no_grad():
                cm(input_ids=ids, use_cache=False, position_ids=pos)

        compiled()
        out["compiled_ms"], out["compiled_min_ms"] = bench(compiled)
        print(f"[b1d] compile(reduce-overhead) = {out['compiled_ms']:.2f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        out["compiled_error"] = f"{type(e).__name__}: {e}"
        print(f"[b1d] compile 失败: {out['compiled_error']}", flush=True)

    if out.get("cudagraph_ms"):
        out["speedup_vs_eager"] = out["eager_ms"] / out["cudagraph_ms"]
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b1d.json")
