"""单条输入的投机解码 profiling：prefill + decode。

回答两个问题：
  1. 投机解码阶段 SM 利用率多少？
  2. draft model 与 target model verify 各占多少时间？

设计要点 —— **完全不改 DeepSpec 框架的代码**，全部用 monkeypatch：
  - `base_evaluator.verify_draft_tokens`  → NVTX "TARGET_VERIFY"
  - `Qwen3Eagle3Evaluator._propose`       → NVTX "DRAFT_PROPOSE"（EAGLE-3 的 γ 次串行 draft）
  - `Qwen3Eagle3Evaluator._update`        → NVTX "DRAFT_UPDATE"（重算 draft hidden / 续 cache）
  - target_model 的**第 1 次** forward     → NVTX "PREFILL"（之后的对 target forward 都在 TARGET_VERIFY 里）

同时在后台线程里以 100ms 间隔采 `nvidia-smi`（带时间戳），
和上面的 phase 起止时间（`time.time()`）对齐，算出**每个阶段的平均 SM 利用率**。

用法（cwd 必须是 DeepSpec-lab 仓库根，才能 import eval）：
    CUDA_VISIBLE_DEVICES=2 python ../profile_single.py
    CUDA_VISIBLE_DEVICES=2 nsys profile -t cuda,nvtx -o ../prof/eagle3_single --force-overwrite=true python ../profile_single.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# 允许从任意 cwd 运行：`import eval` 需要仓库根在 sys.path 上，而 eval.py 的数据集
# 又按 "./eval_datasets" 找，所以还得把 cwd 切到仓库根。
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402

# ---- 固定用一条 gsm8k prompt，decode 步数够看出规律即可 ----
DRAFT = os.environ.get("PROF_DRAFT", "deepseek-ai/eagle3_qwen3_4b_ttt7")
TARGET = os.environ.get("PROF_TARGET", "Qwen/Qwen3-4B")
MAX_NEW_TOKENS = int(os.environ.get("PROF_MAX_NEW_TOKENS", "48"))
OUTDIR = Path(os.environ.get("PROF_OUTDIR", "prof"))

RECORDS: list[dict] = []

# NVTX range 的前缀。预热阶段设成 "WARMUP:"，这样在 nsys 时间线/统计里能把预热
# 和正式测量区分开（两者都会同名前缀，否则混在一起没法过滤）。
LABEL_PREFIX: list[str] = [""]

# GPU 利用率（SM Active）不走 nvidia-smi —— 它的 utilization.gpu 只是「采样周期内
# 至少有一个 kernel 在跑的时间占比」，不是 SM 吞吐，且采样粒度粗（~200ms/次，
# 跨阶段污染）。改用 nsys 的 GPU metrics 时间线，和有同样时间戳的 NVTX range
# 直接对齐（见 analyze_nsys.py）。



# --------------------------------------------------------------------------- #
# NVTX 打桩（monkeypatch，不改框架源码）
# --------------------------------------------------------------------------- #
def _nvtx_push(label: str) -> None:
    torch.cuda.nvtx.range_push(label)


def _nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


def wrap_callable(obj: object, name: str, label: str) -> None:
    """把 obj.name 换成「压 NVTX range + 记录起止时间」的版本。"""
    orig = getattr(obj, name)

    def wrapper(*args, **kwargs):
        full = f"{LABEL_PREFIX[0]}{label}"
        t0 = time.time()
        _nvtx_push(full)
        try:
            return orig(*args, **kwargs)
        finally:
            _nvtx_pop()
            RECORDS.append({"phase": full, "t0": t0, "t1": time.time()})

    setattr(obj, name, wrapper)


def wrap_prefill(target_model: torch.nn.Module) -> dict:
    """target 的**第 1 次** forward 就是 prefill；后续都在 TARGET_VERIFY 范围内。

    返回内部计数器，预热跑完后把它归零（`state["n"] = 0`），否则预热已经用掉了
    "第 1 次"这个名额，正式测量的 prefill 就永远不会被标记。
    """
    orig = target_model.forward
    state = {"n": 0}

    def forward(*args, **kwargs):
        state["n"] += 1
        if state["n"] != 1:
            return orig(*args, **kwargs)
        full = f"{LABEL_PREFIX[0]}PREFILL"
        t0 = time.time()
        _nvtx_push(full)
        try:
            return orig(*args, **kwargs)
        finally:
            _nvtx_pop()
            RECORDS.append({"phase": full, "t0": t0, "t1": time.time()})

    target_model.forward = forward
    return state


# --------------------------------------------------------------------------- #
def main() -> None:
    # eval.py 的 parse_args 只认它自己的参数
    sys.argv = ["eval.py", "--target_name_or_path", TARGET,
                "--draft_name_or_path", DRAFT,
                "--temperature", "1.0",
                "--max-new-tokens", str(MAX_NEW_TOKENS)]

    import deepspec.eval.base_evaluator as BE
    import eval as E
    from transformers import AutoConfig

    # 按 draft checkpoint 自己声明的架构选 evaluator —— 和 eval.py 的
    # EVALUATORS 同一张表。eagle3 与 dspark 的插槽方法名是同一组
    # (_propose / _update)，所以同一套打桩对两者都成立。
    arch = AutoConfig.from_pretrained(DRAFT).architectures[0]
    evaluator_cls = E.EVALUATORS[arch]
    print(f"[prof] arch={arch} evaluator={evaluator_cls.__name__}", flush=True)

    # 先装桩，再建 evaluator（这样 build_models 之后的调用都被覆盖）
    wrap_callable(BE, "verify_draft_tokens", "TARGET_VERIFY")
    wrap_callable(evaluator_cls, "_propose", "DRAFT_PROPOSE")
    wrap_callable(evaluator_cls, "_update", "DRAFT_UPDATE")
    if hasattr(evaluator_cls, "_post_verify"):      # dspark 的置信度头记录
        wrap_callable(evaluator_cls, "_post_verify", "POST_VERIFY")

    args = E.parse_args()
    args.tasks = []          # 不用框架的数据集循环，我们手动喂一条
    evaluator = evaluator_cls(0, args)
    prefill_state = wrap_prefill(evaluator.target_model)

    # DSpark 的置信度头记录器只在 evaluate() 的数据集循环里 start()，我们绕过那层
    # 直接调 generate_one_sample，必须自己 start()，否则 observe() 断言失败。
    # 它的耗时被单独标成 POST_VERIFY —— 属于**测量开销，不是算法本身**，
    # 报数时要和 DRAFT/TARGET 分开看。
    recorder = getattr(evaluator, "confidence_head_recorder", None)
    if recorder is not None:
        recorder.start()
        print("[prof] confidence_head_recorder started (its cost is labelled POST_VERIFY)",
              flush=True)

    from deepspec.data.parser import encode_chat_messages
    from deepspec.eval.base_evaluator import (
        load_and_process_dataset, resolve_stop_token_ids,
    )

    row = load_and_process_dataset("gsm8k")[0]
    input_ids = encode_chat_messages(
        evaluator.tokenizer,
        [{"role": "user", "content": row["turns"][0]}],
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(evaluator.device)
    stop_token_ids = resolve_stop_token_ids(evaluator.target_model, evaluator.tokenizer)
    print(f"[prof] draft={DRAFT} prompt_tokens={input_ids.shape[1]} "
          f"max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 预热：前几次会触发 cuBLAS/cudnn autotune 和 allocator 增长，不计入统计。
    # 预热用的输入复制一份，避免污染下面的统计。
    print("[prof] warmup ...", flush=True)
    LABEL_PREFIX[0] = "WARMUP:"
    evaluator.generate_one_sample(input_ids=input_ids.clone(),
                                  stop_token_ids=stop_token_ids)
    torch.cuda.synchronize()

    RECORDS.clear()             # 丢掉预热阶段的记录
    LABEL_PREFIX[0] = ""        # 预热结束，之后的 range 不再加前缀
    prefill_state["n"] = 0      # 预热已消耗掉「第 1 次 forward」名额，必须归零
    t_start = time.time()
    result = evaluator.generate_one_sample(input_ids=input_ids,
                                           stop_token_ids=stop_token_ids)
    torch.cuda.synchronize()
    t_end = time.time()

    n_out = int(result.output_ids.shape[1]) - int(result.num_input_tokens)
    summary = {
        "draft": DRAFT,
        "target": TARGET,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": n_out,
        "decode_iterations": int(result.verify_count),
        "wall_seconds": t_end - t_start,
        "t_start": t_start,
        "t_end": t_end,
        "phases": RECORDS,
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "phases"}, indent=2))
    print(f"[prof] wrote {OUTDIR}/summary.json (含逐阶段起止时间)", flush=True)


if __name__ == "__main__":
    main()
