"""A 的分析：信念误差能否预测「下一轮」的接受长度？

判决问题
-------
在**控制本轮接受长度 k_t** 之后，本轮的
  - dh   （特征空间信念误差，draft 隐状态 vs target 隐状态）
  - tv   （分布空间信念误差，draft 概率 vs target 概率）
是否仍然预测下一轮的接受长度 k_{t+1}？

对照组是 DSpark 自己的置信度头（conf）。A 有增量的**必要条件**是：
fidelity 的偏相关**强于** conf 的偏相关。否则 DSpark 的头已经把这件事做完了。

统计口径
-------
- 只在同一 prompt 内的相邻轮之间配对（跨 prompt 配对无意义）。
- 偏相关 = 先对 k_t 做线性回归取残差，再算残差间的 Spearman。
  这等价于"控制 k_t 之后"的相关性。
- 显著性用置换检验（打乱 x 的顺序，2000 次），不依赖正态假设。

用法
----
    python scripts/profiling/analyze_fidelity.py a_qwen3_4b.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    # 处理并列
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y 对 x 做线性回归后的残差。"""
    A = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_spearman(y: np.ndarray, x: np.ndarray, ctrl: np.ndarray) -> tuple[float, float]:
    """y ~ x，控制 ctrl。返回 (偏相关, 置换检验 p)。"""
    ry = residuals(y, ctrl)
    rx = residuals(x, ctrl)
    r = spearman(rx, ry)
    if math.isnan(r):
        return float("nan"), float("nan")
    null = []
    for _ in range(2000):
        perm = RNG.permutation(len(rx))
        null.append(spearman(rx[perm], ry))
    null = np.array([v for v in null if not math.isnan(v)])
    p = float((np.abs(null) >= abs(r)).mean()) if len(null) else float("nan")
    return r, p


def load(path: str):
    d = json.loads(Path(path).read_text())
    rounds = d["rounds"]
    # 组内配对：同一 prompt 的 round t 与 t+1
    pairs = []
    by_prompt: dict[int, list[dict]] = {}
    for r in rounds:
        by_prompt.setdefault(r["prompt"], []).append(r)
    for _, rs in sorted(by_prompt.items()):
        for a, b in zip(rs, rs[1:]):
            pairs.append((a, b))
    return d, pairs


def main(path: str) -> None:
    d, pairs = load(path)
    meta = d["meta"]
    print(f"# A 判决：fidelity 能否预测下一轮接受长度")
    print(f"draft={meta['draft']}  dataset={meta['dataset']}  "
          f"prompts={meta['n_prompts']}  pairs={len(pairs)}\n")

    def col(rec, key):
        if key in ("dh", "tv"):
            return rec[key]["mean"]
        return rec.get(key)

    ts, t1 = [], []
    for a, b in pairs:
        vals = [col(a, "dh"), col(a, "tv"), col(a, "conf_mean")]
        if any(v is None for v in vals):
            continue
        if col(a, "accepted") is None or col(b, "accepted") is None:
            continue
        ts.append(a); t1.append(b)

    if len(ts) < 10:
        print(f"样本不足（{len(ts)} 对），无法下结论。"
              f"提高 A_MAX_NEW_TOKENS 或 prompt 数重跑。")
        return

    y_next = np.array([col(b, "accepted") for b in t1], dtype=float)
    k_t = np.array([col(a, "accepted") for a in ts], dtype=float)
    preds = {
        "dh  (特征信念误差)": np.array([col(a, "dh") for a in ts], dtype=float),
        "tv  (分布信念误差)": np.array([col(a, "tv") for a in ts], dtype=float),
        "conf(DSpark 置信度头)": np.array([col(a, "conf_mean") for a in ts], dtype=float),
    }

    print(f"样本: {len(y_next)} 对相邻轮\n")
    print(f"{'预测子':<24}{'裸相关':>10}{'控制k_t后偏相关':>18}{'置换p':>10}")
    print("-" * 62)
    naive = spearman(k_t, y_next)
    print(f"{'k_t (本轮接受长度)':<24}{naive:>10.3f}{'—':>18}{'—':>10}")
    rows = {}
    for name, x in preds.items():
        raw = spearman(x, y_next)
        pr, p = partial_spearman(y_next, x, k_t)
        rows[name] = (raw, pr, p)
        print(f"{name:<24}{raw:>10.3f}{pr:>18.3f}{p:>10.3f}")

    # 分层表：在 k_t 相同/相近的轮里，fidelity 高 vs 低 的下一轮接受长度
    print("\n分层（同一 k_t 上比较 dh 高/低四分位）:")
    print(f"{'k_t':>4}{'n_low':>7}{'n_high':>8}{'k_{t+1} 低dh':>13}{'k_{t+1} 高dh':>13}{'差':>8}")
    dh = preds["dh  (特征信念误差)"]
    for k in sorted(set(k_t.astype(int))):
        m = k_t == k
        if m.sum() < 8:
            continue
        sub_dh, sub_y = dh[m], y_next[m]
        lo, hi = np.quantile(sub_dh, [0.25, 0.75])
        ml, mh = sub_dh <= lo, sub_dh >= hi
        if ml.sum() < 2 or mh.sum() < 2:
            continue
        a_, b_ = sub_y[ml].mean(), sub_y[mh].mean()
        print(f"{k:>4}{ml.sum():>7}{mh.sum():>8}{a_:>13.3f}{b_:>13.3f}{b_ - a_:>8.3f}")

    # ---- 结论 ----
    print("\n" + "=" * 62)
    best = max(rows.items(), key=lambda kv: abs(kv[1][1]) if not math.isnan(kv[1][1]) else -1)
    dh_pr = rows["dh  (特征信念误差)"][1]
    conf_pr = rows["conf(DSpark 置信度头)"][1]
    tv_pr = rows["tv  (分布信念误差)"][1]
    print(f"最强预测子: {best[0]} (偏相关 {best[1][1]:.3f}, p={best[1][2]:.3f})")
    print(f"  dh 偏相关 = {dh_pr:.3f}")
    print(f"  tv 偏相关 = {tv_pr:.3f}")
    print(f"  conf 偏相关 = {conf_pr:.3f}")
    if math.isnan(dh_pr) or math.isnan(conf_pr):
        print("=> 数据不足或方差为零，无法判定。")
    elif abs(dh_pr) > abs(conf_pr) + 0.05 and rows["dh  (特征信念误差)"][2] < 0.05:
        print("=> **A 成立（有增量）**：特征信念误差在控制本轮接受长度后仍显著预测下一轮，")
        print("   且强于 DSpark 置信度头 ⇒ 值得做成 fidelity head + 特征刷新策略。")
    elif rows["dh  (特征信念误差)"][2] < 0.05:
        print("=> **A 部分成立**：fidelity 有预测力，但未明显超过置信度头。")
        print("   增量有限，建议转向「fidelity 用于跨轮/流水场景」而非单轮调度。")
    else:
        print("=> **A 的朴素形式不成立**：信念误差对下一轮接受长度无显著预测力。")
        print("   这是一个干净的负结果，本身可写；但要换机制解释（见 profiling.md 的两个拒绝模式）。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a.json")
