#!/usr/bin/env bash
# 在 8 卡服务器上搭建 DeepSpec-lab 工作区。幂等，可重复运行。
#
# 用法（本地推送执行，不需要在服务器上留副本）：
#   ssh -p 5907 root@120.48.138.220 'bash -s' < server/setup.sh
#
# 只做三件事：探测环境 → clone fork → 建 venv 装依赖。
# 不碰训练、不碰数据管线（那需要 SGLang + 约 38TB target cache，见 README）。
set -euo pipefail

WORKDIR=/mnt/infini/zym-tmp/spec-decode
FORK=https://github.com/YimingZhou2002/DeepSpec-lab.git
UPSTREAM=https://github.com/deepseek-ai/DeepSpec
REPO="$WORKDIR/DeepSpec-lab"

echo "===== 1/3 环境探测 ====="
hostname
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || echo "(nvidia-smi 不可用)"
echo "--- /mnt/infini 空间 ---"
df -h /mnt/infini || true
echo "--- python ---"
python3 -V
echo "--- 工作目录现有内容 ---"
ls -A "$WORKDIR" 2>/dev/null || echo "(目录不存在)"

echo
echo "===== 2/3 clone fork ====="
mkdir -p "$WORKDIR"
if [ -d "$REPO/.git" ]; then
  echo "已存在，跳过 clone：$REPO"
else
  git clone "$FORK" "$REPO"
fi
cd "$REPO"
git remote add upstream "$UPSTREAM" 2>/dev/null || true
git remote -v
git log --oneline -1

echo
echo "===== 3/3 venv + 依赖 ====="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -U pip
# 注意：requirements.txt 顶部注释说 torch 需要装与本机 CUDA 匹配的 wheel。
# 若下面这行装到 CPU 版或版本不匹配的 torch，改用官方 index，例如：
#   pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -q -r requirements.txt

python - <<'PY'
import torch, transformers
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
print("gpu count      :", torch.cuda.device_count())
print("transformers   :", transformers.__version__)
PY

echo
echo "===== 搭建完成 ====="
echo "下一步：评测三个已发布 checkpoint（不碰数据管线）"
cat <<'EOF'
cd /mnt/infini/zym-tmp/spec-decode/DeepSpec-lab && . .venv/bin/activate

# 注意 eval.py 无 --task 参数，固定跑 9 个数据集（共约 3030 条 prompt）
# 结果只打印到 stdout，所以必须 tee 到文件
for draft in \
  deepseek-ai/eagle3_qwen3_4b_ttt7 \
  deepseek-ai/dflash_qwen3_4b_block7 \
  deepseek-ai/dspark_qwen3_4b_block7 ; do
  name=$(basename "$draft")
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path "$draft" \
    --temperature 1.0 \
    2>&1 | tee "results_${name}.log"
done
EOF
