"""B1b: batch 扩展 —— 判定「24ms/前向」到底是带宽受限还是延迟受限。

判据
----
一次前向要读 7.49 GiB 的 target 权重（与 batch / token 数无关）。

- 若耗时**随 batch 基本不变** ⇒ 权重流主导，算力/延迟有巨大余量
  ⇒ 单条序列根本填不满 GPU，投机解码在 bs=1 是纯粹的「延迟掩盖」问题
- 若耗时**随 batch 线性增长** ⇒ 每序列的算力/延迟才是瓶颈

这决定了 B 的机制解释，也决定了「verify 一次能顺带服务多少条序列」。

用法
----
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/b1b_batch_scaling.py out.json
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

TARGET = os.environ.get("B1B_TARGET", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("B1B_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
CONTEXTS = [int(x) for x in os.environ.get("B1B_CONTEXTS", "92,2048").split(",")]
BATCHES = [int(x) for x in os.environ.get("B1B_BATCHES", "1,2,4,8,16,32,64").split(",")]
NS = [int(x) for x in os.environ.get("B1B_NS", "1,8").split(",")]
REPS = int(os.environ.get("B1B_REPS", "5"))


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
    weight_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    print(json.dumps({"weight_gib": round(weight_gib, 3),
                      "gpu": torch.cuda.get_device_name(0),
                      "props_mem_clock_note": "见 nvidia-smi 输出"}, indent=2), flush=True)

    results = {"meta": {"target": TARGET, "weight_gib": weight_gib,
                        "gpu": torch.cuda.get_device_name(0)}, "sweep": []}
    max_len = max(CONTEXTS) + max(NS) + 8
    base_ids = torch.randint(0, cfg.vocab_size, (1, max_len), device=dev)

    for L in CONTEXTS:
        for B in BATCHES:
            ctx = base_ids.repeat(B, 1)
            for N in NS:
                nid = ctx[:, L:L + N]
                pos = torch.arange(L, L + N, device=dev)[None].expand(B, -1)

                def run():
                    cache = DynamicCache()
                    with torch.no_grad():
                        o = model(input_ids=ctx[:, :L], past_key_values=cache,
                                  use_cache=True,
                                  position_ids=torch.arange(L, device=dev)[None].expand(B, -1))
                    cache = o.past_key_values
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    with torch.no_grad():
                        model(input_ids=nid, past_key_values=cache, use_cache=True,
                              position_ids=pos)
                    e.record()
                    torch.cuda.synchronize()
                    return s.elapsed_time(e)

                run()
                ts = [run() for _ in range(REPS)]
                med = statistics.median(ts)
                row = {
                    "ctx_len": L, "batch": B, "N": N,
                    "ms": med,
                    "ms_per_seq": med / B,
                    "seq_per_s": B / (med / 1e3),
                    "effective_weight_bw_GBs": (weight_gib * 2**30 / 1e9) / (med / 1e3),
                }
                results["sweep"].append(row)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in row.items()}), flush=True)

    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[b1b] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "b1b.json")
