"""B1: verify 成本对「被验证 token 数 N」的标度律。

要回答的问题
-----------
投机解码每次迭代 = draft 提案 + target 验证。verify 做的是**一次前向**，输入是
[anchor, draft_1, ..., draft_N]（长度 N+1），对 KV cache 做带 mask 的前向。

关键问题：**verify 的耗时是否随 N 增长？**

- 若随 N 显著增长  ⇒ verify 是「每 token 一份成本」，γ 越大越贵
- 若近似平坦      ⇒ 瓶颈是「权重流 + 每步固定开销」，**与验证几个 token 基本无关**
                      ⇒ 现有 drafter 的 γ=7 可能远小于最优，加长提案几乎免费

计时口径（关键）
--------------
只用 **CUDA event 包住那一次 N-token 前向**。建 KV cache（prefill）在事件窗口**之外**，
每个 rep 重建一次但不计时 —— 否则长 context 的 prefill（秒级）会把 N-token 前向
（几十 ms）的信号彻底淹没。这是本脚本最容易写错的地方。

额外测 SERIAL_N：把 N 个 token 拆成 N 次单 token 前向。
它与 ONE_SHOT_N 的差值 = **每步固定开销**，是 B 成本模型里最关键的一项。

用法
----
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/b1_verify_scaling.py out.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

TARGET = os.environ.get("B1_TARGET", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("B1_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
CONTEXTS = [int(x) for x in os.environ.get("B1_CONTEXTS", "92,2048,8192").split(",")]
NS = [int(x) for x in os.environ.get("B1_NS", "1,2,4,8,16,32,64,128").split(",")]
REPS = int(os.environ.get("B1_REPS", "7"))


def timed(fn, reps=REPS):
    """返回 (median_ms, min_ms)。fn 内部应只包含被计时的 GPU 工作。"""
    fn()  # warmup
    ts = []
    for _ in range(reps):
        ts.append(fn())
    return statistics.median(ts), min(ts)


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

    n_params = sum(p.numel() for p in model.parameters())
    weight_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    meta = {
        "target": TARGET, "arch": arch, "dtype": str(next(model.parameters()).dtype),
        "attn": str(getattr(cfg, "_attn_implementation", None)),
        "n_params": n_params, "weight_gib": round(weight_gib, 3),
        "n_layers": cfg.num_hidden_layers,
        "kv_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(cfg, "head_dim", None),
    }
    print(json.dumps(meta, indent=2), flush=True)

    def prefill(ids):
        cache = DynamicCache()
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(ids.shape[1], device=dev)[None])
        return out.past_key_values

    def make_oneshot(ctx, L, N):
        """一次前向 N 个 token —— verify 的真实形态。只计时这一次前向。"""
        nid = ctx[:, L:L + N]
        pos = torch.arange(L, L + N, device=dev)[None]

        def run():
            cache = prefill(ctx[:, :L])              # 不计时
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            with torch.no_grad():
                model(input_ids=nid, past_key_values=cache, use_cache=True, position_ids=pos)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        return run

    def make_serial(ctx, L, N):
        """N 次单 token 前向 —— 暴露每步固定开销。"""
        def run():
            cache = prefill(ctx[:, :L])
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            with torch.no_grad():
                for i in range(N):
                    o = model(input_ids=ctx[:, L + i:L + i + 1], past_key_values=cache,
                              use_cache=True,
                              position_ids=torch.full((1, 1), L + i, device=dev))
                    cache = o.past_key_values
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        return run

    results = {"meta": meta, "sweep": [], "prefill": []}
    max_len = max(CONTEXTS) + max(NS) + 8

    for L in CONTEXTS:
        ids = torch.randint(0, cfg.vocab_size, (1, max_len), device=dev)
        for N in NS:
            med, mn = timed(make_oneshot(ids, L, N))
            row = {"ctx_len": L, "N": N, "one_shot_ms": med, "one_shot_min_ms": mn,
                   "per_token_ms": med / N}
            if N <= 16:
                ser, _ = timed(make_serial(ids, L, N), reps=5)
                row["serial_ms"] = ser
                if N > 1:
                    row["per_step_overhead_ms"] = (ser - med) / (N - 1)
            results["sweep"].append(row)
            print(json.dumps(row), flush=True)

    for L in [64, 256, 1024, 4096, 16384]:
        ids = torch.randint(0, cfg.vocab_size, (1, L), device=dev)
        pos = torch.arange(L, device=dev)[None]

        def run():
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, position_ids=pos)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)

        med, _ = timed(run, reps=3)
        row = {"ctx_len": L, "prefill_ms": med, "per_token_ms": med / L}
        results["prefill"].append(row)
        print(json.dumps(row), flush=True)

    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[b1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b1.json")
