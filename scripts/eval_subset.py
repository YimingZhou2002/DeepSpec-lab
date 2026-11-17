"""跑 eval.py 的任意数据集子集（eval.py 本身没有 --task 参数，固定跑全部 9 个）。

快速迭代用：验证改动是否破坏了链路、或只看某几个数据集，不必等全量 3030 条。

用法（仓库根目录下）：
    # 全卡
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/eval_subset.py \
        gsm8k 200 Qwen/Qwen3-4B deepseek-ai/dspark_qwen3_4b_block7 1.0
    # 单卡冒烟
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_subset.py \
        gsm8k 4 Qwen/Qwen3-4B deepseek-ai/dspark_qwen3_4b_block7 1.0

参数：<dataset> <max_samples> <target> <draft> <temperature>

注意：eval.py 的 main() 里 nprocs = torch.cuda.device_count()，且 init_dist 取
world_size = WORLD_SIZE(默认1) × device_count() —— 所以可见几张卡就用几张卡做数据分片，
指标是跨 rank all_reduce 的，口径与全量跑一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 允许从任意 cwd 运行：eval.py 在仓库根，且它按 "./eval_datasets" 找数据集
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402

import eval as E  # noqa: E402


def run(dataset: str, limit: int, target: str, draft: str, temp: str) -> None:
    # 清掉自定义位置参数，否则 eval.py 的 argparse 会报错
    sys.argv = ["eval.py", "--target_name_or_path", target,
                "--draft_name_or_path", draft, "--temperature", temp]
    # parse_args() 会执行 args.tasks = list(TASKS)；必须在它之前改掉模块级 TASKS
    E.TASKS = [(dataset, limit)]
    args = E.parse_args()
    torch.multiprocessing.spawn(E.main, args=(args,), nprocs=torch.cuda.device_count())


if __name__ == "__main__":
    # spawn 会在子进程里用 runpy 重跑本模块（__name__ == "__mp_main__"），
    # 所以 argv 解析必须放在这个保护之下，否则 sys.argv 已被替换会解析崩溃。
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
