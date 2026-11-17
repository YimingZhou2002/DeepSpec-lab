"""A: 特征保真度（feature fidelity）探针 —— 回答 A 的前提是否成立。

A 的主张
-------
投机解码的正确性单位一直是 **token**。但现代 drafter 越来越多地在**自己预测的特征**
上做条件（DSpine 逐层注入 predecessor 的 predicted feature、DSpark 的 Markov head、
DFlash 的块内并行）。于是有一个从未被验证的失败模式：

    **token 被接受了，但 drafter 当时的"信念"是错的。**

本脚本要做的判决性测量：这种信念误差是否**因果地**预测**下一轮**的接受长度。

为什么定义明确（关键）
--------------------
- 对齐关系：draft 的 `block_hidden[:, i, :]` 预测 draft token i；
  target 在 verify 里第 i 个位置的隐状态**也**预测 draft token i。⇒ **逐位置 i↔i 对齐**。
- 两者可比的依据：draft 的 `lm_head` 在初始化时从 target 复制
  （modeling.py:281 `self.lm_head.weight.copy_(lm_head.weight.detach())`），
  且 hidden_size 相同 ⇒ 同一个特征空间。
- **标签免费**：verify 返回的 target 隐状态就是 drafter 预测隐状态的真值。
  不需要任何额外前向。

同时采两组指标（都是零额外算力）
------------------------------
1. **特征空间**：dh_i = ‖h_draft_i − h_target_i‖ / ‖h_target_i‖
2. **分布空间**：tv_i = ½·Σ|p_target_i − p_draft_i|（两者在 verify 里本来就算了）

以及对照组：**DSpark 自己的置信度头**在同一轮给出的接受概率。
A 的成立条件是 fidelity 对"下一轮接受长度"的预测力**优于**置信度头 ——
否则 DSpark 的头已经把这件事做完了，A 就没有增量。

用法
----
    CUDA_VISIBLE_DEVICES=0 python scripts/profiling/a_feature_fidelity.py out.json 5
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

TARGET = os.environ.get("A_TARGET", "Qwen/Qwen3-4B")
DRAFT = os.environ.get("A_DRAFT", "deepseek-ai/dspark_qwen3_4b_block7")
DATASET = os.environ.get("A_DATASET", "gsm8k")
MAX_NEW_TOKENS = int(os.environ.get("A_MAX_NEW_TOKENS", "256"))

PENDING: dict = {}   # 本轮 draft 侧记录，等 verify 来配对
ROUNDS: list[dict] = []
PROMPT_IDX = [0]     # 当前 prompt 序号，写进每条 round，避免分析时跨 prompt 配对


def _stats(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"mean": None, "max": None, "n": 0}
    return {"mean": float(x.mean()), "max": float(x.max()), "n": int(x.numel())}


def main(out_path: str, n_prompts: int) -> None:
    sys.argv = ["eval.py", "--target_name_or_path", TARGET,
                "--draft_name_or_path", DRAFT, "--temperature", "1.0",
                "--max-new-tokens", str(MAX_NEW_TOKENS)]
    import eval as E  # noqa: E402
    import deepspec.eval.base_evaluator as BE  # noqa: E402
    from deepspec.eval.dspark import draft_ops as DO  # noqa: E402

    arch = AutoConfig.from_pretrained(DRAFT).architectures[0]
    evaluator_cls = E.EVALUATORS[arch]
    print(f"[a] arch={arch} evaluator={evaluator_cls.__name__}", flush=True)

    # ---- 插桩 1：draft 的块隐状态 ---------------------------------------- #
    orig_block = DO.forward_dspark_draft_block

    def block(*args, **kwargs):
        out = orig_block(*args, **kwargs)
        PENDING["draft_h"] = out.detach().float()[0]          # [block, H]
        return out

    # 关键：evaluator.py 用的是 `from ...draft_ops import forward_dspark_draft_block`，
    # 名字已经绑进 evaluator 模块自己的命名空间。只改 draft_ops 是**不生效**的，
    # 必须两个命名空间都换掉。
    import deepspec.eval.dspark.evaluator as DE
    DO.forward_dspark_draft_block = block
    if hasattr(DE, "forward_dspark_draft_block"):
        DE.forward_dspark_draft_block = block
    print(f"[a] patched draft block: draft_ops=1 evaluator={hasattr(DE, 'forward_dspark_draft_block')}", flush=True)

    # ---- 插桩 2：draft 的 proposal（拿 draft_probs 与置信度） ------------- #
    orig_propose = evaluator_cls._propose

    def propose(self, *args, **kwargs):
        prop = orig_propose(self, *args, **kwargs)
        PENDING["draft_probs"] = prop.draft_probs.detach().float()[0] if prop.draft_probs is not None else None
        cl = getattr(prop, "confidence_logits", None)
        PENDING["conf"] = torch.sigmoid(cl.detach().float()[0]) if cl is not None else None
        PENDING["n_prop"] = int(prop.draft_token_count)
        return prop

    evaluator_cls._propose = propose

    # ---- 插桩 3：verify —— 配对并算指标 ---------------------------------- #
    orig_verify = BE.verify_draft_tokens

    def verify(*args, **kwargs):
        res = orig_verify(*args, **kwargs)
        draft_h = PENDING.pop("draft_h", None)
        draft_p = PENDING.pop("draft_probs", None)
        conf = PENDING.pop("conf", None)
        n_prop = PENDING.pop("n_prop", 0)

        if draft_h is None or res.accepted_draft_tokens == 0:
            if n_prop:
                ROUNDS.append({"prompt": PROMPT_IDX[0], "accepted": int(res.accepted_draft_tokens), "n_prop": n_prop,
                               "dh": _stats(torch.empty(0)), "tv": _stats(torch.empty(0)),
                               "conf_mean": None, "note": "no_draft_hidden_or_zero_accept"})
            return res

        tgt_h = res.target_output.hidden_states[-1].detach().float()[0]   # [k+1, H]
        tgt_p = res.target_probs.detach().float()[0]                      # [k+1, V]
        k = int(res.accepted_draft_tokens)

        # 对齐：draft 位置 i ↔ target 位置 i（都预测 draft token i）
        n = min(k, draft_h.shape[0], tgt_h.shape[0] - 0)
        d_h = draft_h[:n]
        t_h = tgt_h[:n]
        dh = (d_h - t_h).norm(dim=-1) / t_h.norm(dim=-1).clamp_min(1e-6)

        tv = torch.empty(0)
        if draft_p is not None and draft_p.shape[0] >= n:
            dp = draft_p[:n]
            tp = tgt_p[:n]
            tv = 0.5 * (tp - dp).abs().sum(dim=-1)

        ROUNDS.append({
            "prompt": PROMPT_IDX[0],
            "accepted": k,
            "n_prop": n_prop,
            "dh": _stats(dh),
            "tv": _stats(tv),
            "conf_mean": float(conf[:n].mean()) if conf is not None and conf.numel() >= n else None,
        })
        return res

    BE.verify_draft_tokens = verify

    # ---- 跑若干条 prompt -------------------------------------------------- #
    args = E.parse_args()
    args.tasks = []
    evaluator = evaluator_cls(0, args)
    # 绕过了 evaluate() 的数据集循环，置信度头记录器必须自己 start()，
    # 否则 _post_verify -> observe() 会撞上 confidence_head.py:351 的断言。
    _rec = getattr(evaluator, "confidence_head_recorder", None)
    if _rec is not None:
        _rec.start()
    from deepspec.eval.base_evaluator import load_and_process_dataset, resolve_stop_token_ids
    from deepspec.data.parser import encode_chat_messages

    rows = load_and_process_dataset(DATASET)[:n_prompts]
    stop = resolve_stop_token_ids(evaluator.target_model, evaluator.tokenizer)
    print(f"[a] dataset={DATASET} prompts={len(rows)} max_new={MAX_NEW_TOKENS}", flush=True)

    per_prompt = []
    for pi, row in enumerate(rows):
        PROMPT_IDX[0] = pi
        ids = encode_chat_messages(evaluator.tokenizer,
                                   [{"role": "user", "content": row["turns"][0]}],
                                   add_generation_prompt=True,
                                   enable_thinking=False).to(evaluator.device)
        before = len(ROUNDS)
        t0 = time.time()
        res = evaluator.generate_one_sample(input_ids=ids, stop_token_ids=stop)
        dt = time.time() - t0
        seg = ROUNDS[before:]
        n_out = int(res.output_ids.shape[1]) - int(res.num_input_tokens)
        per_prompt.append({"prompt": pi, "rounds": len(seg), "generated": n_out,
                           "verify_count": int(res.verify_count), "seconds": dt})
        print(f"[a] prompt {pi}: rounds={len(seg)} gen={n_out} "
              f"verify={res.verify_count} {dt:.1f}s", flush=True)

    out = {"meta": {"target": TARGET, "draft": DRAFT, "dataset": DATASET,
                    "max_new_tokens": MAX_NEW_TOKENS, "n_prompts": n_prompts},
           "per_prompt": per_prompt, "rounds": ROUNDS}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"[a] wrote {out_path}  ({len(ROUNDS)} rounds)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a.json",
         int(sys.argv[2]) if len(sys.argv) > 2 else 5)
