"""用 nsys 的 sqlite 导出，把 NVTX 阶段和 GPU metrics 时间线对齐。

回答：投机解码各阶段（PREFILL / DRAFT_PROPOSE / DRAFT_UPDATE / TARGET_VERIFY）
的 **SM 利用率** 是多少。

口径说明（重要）：
  - 用 `SMs Active [Throughput %]`：平均有多少比例的 SM 处于活跃状态。
    batch=1 的 decode 只有极少数 SM 有活干，所以这个值天然偏低 —— 这正是
    "投机解码是 memory-bound、GPU 利用率上不去" 的量化体现。
  - 同时报 Tensor Active / SM Issue / DRAM 带宽，用来判断瓶颈在算力还是在访存。
  - 不用 nvidia-smi 的 utilization.gpu（那只是「有 kernel 在跑的时间占比」，
    粒度粗且跨阶段污染）。

用法：
    python analyze_nsys.py prof/eagle3/nsys_eagle3.sqlite
"""

from __future__ import annotations

import bisect
import sqlite3
import sys
from collections import defaultdict

# metricId -> 显示名（来自 TARGET_INFO_GPU_METRICS）
METRICS = {
    7: "SM Active",
    9: "Tensor Active",
    8: "SM Issue",
    6: "GR Active",
    19: "DRAM Rd",
    20: "DRAM Wr",
}

MEASURED_PHASES = ("PREFILL", "DRAFT_PROPOSE", "DRAFT_UPDATE", "TARGET_VERIFY")


def load_series(con: sqlite3.Connection, metric_id: int):
    rows = con.execute(
        "SELECT timestamp, value FROM GPU_METRICS WHERE metricId = ? ORDER BY timestamp",
        (metric_id,),
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def pool(ts: list[int], vs: list[float], windows: list[tuple[int, int]]):
    """把若干时间窗内的样本汇到一起，返回 (均值, 样本数)。

    按样本池化而不是「先算每个窗口的均值再平均」—— 后者会被短窗口过度加权。
    """
    picked: list[float] = []
    for t0, t1 in windows:
        i = bisect.bisect_left(ts, t0)
        j = bisect.bisect_right(ts, t1)
        picked.extend(vs[i:j])
    if not picked:
        return float("nan"), 0
    return sum(picked) / len(picked), len(picked)


def main(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text IS NOT NULL"
    ).fetchall()

    windows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in rows:
        windows[text].append((start, end))

    series = {mid: load_series(con, mid) for mid in METRICS}

    # 有效采样间隔（透明起见报出来）
    ref_ts = series[7][0]
    if len(ref_ts) > 1:
        span = ref_ts[-1] - ref_ts[0]
        eff_us = span / (len(ref_ts) - 1) / 1000.0
    else:
        eff_us = float("nan")

    head = (f"{'phase':<16}{'runs':>6}{'NVTX ms':>10}{'avg ms':>9}"
            + "".join(f"{METRICS[m]:>14}" for m in METRICS))
    print(f"SM 利用率口径: nsys GPU metrics, 有效采样间隔 ≈ {eff_us:.0f} us")
    print(head)
    print("-" * len(head))

    measured_ms: dict[str, float] = {}
    for phase in MEASURED_PHASES:
        ws = windows.get(phase, [])
        if not ws:
            continue
        total_ns = sum(e - s for s, e in ws)
        measured_ms[phase] = total_ns / 1e6
        cells = []
        for mid in METRICS:
            mean, n = pool(*series[mid], ws)
            cells.append(f"{mean:>13.1f}%" if n else f"{'-':>14}")
        print(f"{phase:<16}{len(ws):>6}{total_ns / 1e6:>10.1f}"
              f"{total_ns / 1e6 / len(ws):>9.2f}" + "".join(cells))

    # 非测量的（预热 / 其它）也报一下，便于核对
    for text, ws in sorted(windows.items()):
        if text in MEASURED_PHASES:
            continue
        total_ns = sum(e - s for s, e in ws)
        if total_ns / 1e6 < 1.0:
            continue
        print(f"{text:<16}{len(ws):>6}{total_ns / 1e6:>10.1f}"
              f"{total_ns / 1e6 / len(ws):>9.2f}" + " " * (14 * len(METRICS)))

    # ---- 只统计 decode 循环（不含 prefill）----
    loop = {p: measured_ms.get(p, 0.0)
            for p in ("DRAFT_PROPOSE", "DRAFT_UPDATE", "TARGET_VERIFY")}
    loop_total = sum(loop.values())
    if loop_total <= 0:
        return

    draft_ms = loop["DRAFT_PROPOSE"] + loop["DRAFT_UPDATE"]
    verify_ms = loop["TARGET_VERIFY"]

    print("\n=== decode 循环：时间分配（NVTX 侧）===")
    for name, ms in sorted(loop.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16}{ms:>9.1f} ms{100 * ms / loop_total:>8.1f}%")
    print(f"  {'DRAFT 合计':<16}{draft_ms:>9.1f} ms{100 * draft_ms / loop_total:>8.1f}%")
    print(f"  draft : verify = {draft_ms / verify_ms:.3f}")

    print("\n=== decode 循环：SM 利用率 ===")
    loop_windows = [w for p in loop for w in windows.get(p, [])]
    for mid in METRICS:
        mean, n = pool(*series[mid], loop_windows)
        print(f"  {METRICS[mid]:<14}{mean:>7.1f}%   (samples={n})")

    # draft 与 verify 分别的 SM 利用率 —— 用户最关心的对比
    print("\n=== draft vs verify 的 SM 利用率对比 ===")
    dw = [w for p in ("DRAFT_PROPOSE", "DRAFT_UPDATE") for w in windows.get(p, [])]
    for label, ws in (("draft", dw), ("target verify", windows.get("TARGET_VERIFY", []))):
        parts = []
        for mid in METRICS:
            mean, _ = pool(*series[mid], ws)
            parts.append(f"{METRICS[mid]}={mean:.1f}%")
        print(f"  {label:<14}" + "  ".join(parts))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prof/eagle3/nsys_eagle3.sqlite")
