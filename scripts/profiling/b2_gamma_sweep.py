"""B2: 真实投机解码下的 γ 扫描 —— 最优提案长度在哪里？

B1 已证明：**target 的一次前向成本与验证几个 token 基本无关**（128× token 只贵 12-18%），
成本由「把 7.49 GiB target 权重流一遍」决定，≈ 24-25 ms/次前向。

于是效率的唯一杠杆变成「每次 target 前向能兑现多少 token」= τ。本脚本测：
**把 γ 调大，τ 涨得比成本快吗？** 即真实的最优 γ 是否远大于现有 checkpoint 的 7。

γ 注入点（现有 checkpoint，无需重训）
------------------------------------
- EAGLE-3:  `draft_model.ttt_length`      （_propose 里 for _ in range(max_proposal_tokens)）
- DFlash/DSpark: `draft_model.block_size`  （max_proposal_tokens 是读它的 property）

超出训练时的 γ 会退化，这正是要测的：τ 是否还在涨。曲线形状本身也是结论。

同时累计逐位置接受率，看接受曲线随 γ 的变化。

用法
----
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/b2_gamma_sweep.py <draft> out.json [n_prompts]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

TARGET = os.environ.get("B2_TARGET", "Qwen/Qwen3-4B")
DATASET = os.environ.get("B2_DATASET", "gsm8k")
MAX_NEW_TOKENS = int(os.environ.get("B2_MAX_NEW_TOKENS", "512"))
GAMMAS = [int(x) for x in os.environ.get("B2_GAMMAS", "1,2,3,5,7,10,13,16,24,32").split(",")]

PHASES: list[dict] = []
ACCEPT: list[int] = []      # 每轮接受的 draft token 数
POS_ACC: list[list[int]] = []  # 每轮的逐位置接受标记


def main(draft: str, out_path: str, n_prompts: int) -> None:
    sys.argv = ["eval.py", "--target_name_or_path", TARGET,
                "--draft_name_or_path", draft, "--temperature", "1.0",
                "--max-new-tokens", str(MAX_NEW_TOKENS)]
    import eval as E  # noqa: E402
    import deepspec.eval.base_evaluator as BE  # noqa: E402
    from deepspec.eval.base_evaluator import (  # noqa: E402
        load_and_process_dataset, resolve_stop_token_ids,
    )
    from deepspec.data.parser import encode_chat_messages  # noqa: E402

    arch = AutoConfig.from_pretrained(draft).architectures[0]
    evaluator_cls = E.EVALUATORS[arch]
    print(f"[b2] arch={arch} draft={draft}", flush=True)

    # ---- 阶段计时：只记 CPU 侧起止，用于把 total 拆成 verify / draft ------- #
    def wrap_callable(obj, name, label):
        orig = getattr(obj, name)

        def wrapper(*a, **kw):
            t0 = time.time()
            try:
                return orig(*a, **kw)
            finally:
                PHASES.append({"phase": label, "dt": time.time() - t0})

        setattr(obj, name, wrapper)

    wrap_callable(BE, "verify_draft_tokens", "TARGET_VERIFY")
    wrap_callable(evaluator_cls, "_propose", "DRAFT_PROPOSE")
    if hasattr(evaluator_cls, "_update"):
        wrap_callable(evaluator_cls, "_update", "DRAFT_UPDATE")

    orig_verify = BE.verify_draft_tokens

    def verify(*a, **kw):
        res = orig_verify(*a, **kw)
        ACCEPT.append(int(res.accepted_draft_tokens))
        m = res.accept_prefix_mask
        POS_ACC.append(m[0].tolist() if m is not None else [])
        return res

    BE.verify_draft_tokens = verify

    args = E.parse_args()
    args.tasks = []
    evaluator = evaluator_cls(0, args)
    # 本扫描里直接把置信度头记录器**关掉**：
    #   1) 它内部把 init 时的 block_size 存了下来，改 γ 之后会撞
    #      confidence_head.py:73 的 `assert pos_count <= self.block_size`；
    #   2) 它的耗时是**测量开销不是算法**（见 docs/baselines/profiling.md 里单列的
    #      POST_VERIFY），留着会污染 draft/verify 的拆分。
    # DSpark 的 _post_verify 开头就是 `if self.confidence_head_recorder is None: return`，
    # 所以置 None 是安全短路。
    if getattr(evaluator, "confidence_head_recorder", None) is not None:
        evaluator.confidence_head_recorder = None
        print("[b2] confidence_head_recorder 已关闭（纯计时口径）", flush=True)

    # γ 的注入点
    if hasattr(evaluator.draft_model, "ttt_length"):
        knob, orig_gamma = "ttt_length", int(evaluator.draft_model.ttt_length)
    elif hasattr(evaluator.draft_model, "block_size"):
        knob, orig_gamma = "block_size", int(evaluator.draft_model.block_size)
    else:
        raise RuntimeError("找不到 γ 注入点（ttt_length / block_size）")
    print(f"[b2] gamma knob = draft_model.{knob}, 原始值={orig_gamma}", flush=True)

    rows = load_and_process_dataset(DATASET)[:n_prompts]
    stop = resolve_stop_token_ids(evaluator.target_model, evaluator.tokenizer)
    prompts = [
        encode_chat_messages(evaluator.tokenizer,
                             [{"role": "user", "content": r["turns"][0]}],
                             add_generation_prompt=True, enable_thinking=False
                             ).to(evaluator.device)
        for r in rows
    ]
    print(f"[b2] dataset={DATASET} prompts={len(prompts)} max_new={MAX_NEW_TOKENS}", flush=True)

    results = {"meta": {"target": TARGET, "draft": draft, "arch": arch,
                        "dataset": DATASET, "n_prompts": len(prompts),
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "gamma_knob": f"draft_model.{knob}", "orig_gamma": orig_gamma},
               "gammas": []}

    for g in GAMMAS:
        setattr(evaluator.draft_model, knob, g if knob == "ttt_length" else g)
        PHASES.clear(); ACCEPT.clear(); POS_ACC.clear()
        gen = 0; vc = 0; t0 = time.time()
        for ids in prompts:
            r = evaluator.generate_one_sample(input_ids=ids.clone(), stop_token_ids=stop)
            gen += int(r.output_ids.shape[1]) - int(r.num_input_tokens)
            vc += int(r.verify_count)
        torch.cuda.synchronize()
        dt = time.time() - t0

        agg: dict[str, float] = {}
        for p in PHASES:
            agg[p["phase"]] = agg.get(p["phase"], 0.0) + p["dt"]
        tau = gen / max(vc, 1)
        row = {
            "gamma": g,
            "generated": gen, "verify_count": vc,
            "tau": tau,
            "tokens_per_sec": gen / dt,
            "wall_s": dt,
            "TARGET_VERIFY_ms": agg.get("TARGET_VERIFY", 0.0) * 1e3,
            "DRAFT_PROPOSE_ms": agg.get("DRAFT_PROPOSE", 0.0) * 1e3,
            "DRAFT_UPDATE_ms": agg.get("DRAFT_UPDATE", 0.0) * 1e3,
            "mean_accepted": sum(ACCEPT) / max(len(ACCEPT), 1),
        }
        # 逐位置接受率
        n_pos = max((len(x) for x in POS_ACC), default=0)
        pos = []
        for j in range(n_pos):
            seen = [x[j] for x in POS_ACC if len(x) > j]
            pos.append(sum(seen) / len(seen) if seen else None)
        row["accept_rate_by_pos"] = pos
        results["gammas"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "accept_rate_by_pos"}), flush=True)
        print("   pos_acc=" + json.dumps([None if v is None else round(v, 3) for v in pos[:16]]), flush=True)

    setattr(evaluator.draft_model, knob, orig_gamma)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[b2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
