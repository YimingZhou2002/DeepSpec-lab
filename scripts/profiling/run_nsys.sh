#!/usr/bin/env bash
# 用 nsys 采集投机解码的单条输入（prefill + decode），三个 draft 依次跑。
#
# 用法（在仓库根目录执行）：
#   bash scripts/profiling/run_nsys.sh                    # 跑三个默认 draft
#   DRAFTS="deepseek-ai/eagle3_qwen3_4b_ttt7" bash scripts/profiling/run_nsys.sh
#
# 前置：
#   1. nsys >= 2023.1（GPU metrics 是那时引入的）。apt 里 Ubuntu multiverse 的
#      nsight-systems 只有 2022.4，**没有 --gpu-metrics-devices**，要从 NVIDIA
#      devtools 源装：https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/
#   2. export NSYS_TMPDIR=<大分区路径>   # nsys 需要 >200MB 临时空间，/tmp 常常不够
#
# 采集完用 scripts/profiling/analyze_nsys.py 出阶段表。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROF_ROOT="${PROF_ROOT:-$REPO_ROOT/prof}"
GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DRAFTS="${DRAFTS:-deepseek-ai/eagle3_qwen3_4b_ttt7 deepseek-ai/dflash_qwen3_4b_block7 deepseek-ai/dspark_qwen3_4b_block7}"

export NSYS_TMPDIR="${NSYS_TMPDIR:?请设置 NSYS_TMPDIR 到一个有 >200MB 空间的目录}"
mkdir -p "$NSYS_TMPDIR" "$PROF_ROOT"

for draft in $DRAFTS; do
  tag="$(basename "$draft")"
  out="$PROF_ROOT/$tag"
  mkdir -p "$out"
  echo "########## $tag ##########"

  # CUDA_VISIBLE_DEVICES 只留一张卡；--gpu-metrics-devices=cuda-visible 让 nsys
  # 只采这张卡（用 all 会去采其它卡，且在 CUDA_VISIBLE_DEVICES 下语义容易搞混）
  CUDA_VISIBLE_DEVICES="$GPU" \
  PROF_DRAFT="$draft" PROF_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" PROF_OUTDIR="$out" \
    nsys profile --trace=cuda,nvtx \
      --gpu-metrics-devices=cuda-visible \
      --sample=none --cpuctxsw=none \
      --force-overwrite=true \
      -o "$out/nsys_$tag" \
      python "$REPO_ROOT/scripts/profiling/profile_single.py" > "$out/run.log" 2>&1
  echo "  profile rc=$?"

  nsys export --type sqlite --force-overwrite=true \
    -o "$out/nsys_$tag.sqlite" "$out/nsys_$tag.nsys-rep" > /dev/null 2>&1
  echo "  export rc=$?"

  grep -E "arch=|generated_tokens|decode_iterations|wall_seconds" "$out/run.log" | head -5
  python "$REPO_ROOT/scripts/profiling/analyze_nsys.py" "$out/nsys_$tag.sqlite" || true
done

echo "NSYS ALL DONE"
