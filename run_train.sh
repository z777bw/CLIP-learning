#!/usr/bin/env bash
# ============================================================================
# 单机多卡训练启动脚本（使用 torchrun）—— 一键启动
#
# 用法：
#   bash run_train.sh                    # 按 configs/clip_vit_b32.yaml 训练
#   bash run_train.sh epochs=10          # 任意 Hydra 字段都能临时覆盖
#   bash run_train.sh batch_size=128 lr=3.0e-4
#
# 超参数统一放在 configs/clip_vit_b32.yaml（已按 118K 图文对 / 2 卡标定），
# 本脚本只负责「环境相关」的三件事：挂哪几张卡、读哪份数据、结果写到哪。
# 这样直接 `torchrun train/main.py data_root=...` 也能拿到同一套超参，
# 不会出现「脚本里一套、配置文件里另一套」的漂移。
#
# 数据格式：
#   本次使用解压后的 COCO 扁平目录（<data_root>/{train,test}/s0000000.jpg + .txt）。
#   留空 data_format 会自动推断成 coco_flat；若改用 JSONL 清单，把下面换成
#   data_manifest=/path/to/train.jsonl 并把 data_format 显式设为 manifest。
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 指定使用哪几张 GPU（索引从 0 开始）。
# 注意：2/3/5 号卡被其它进程占用，所以这里只挂了空闲的几张。
# ---------------------------------------------------------------------------
GPUS="0,1"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# nproc_per_node = 卡数（由 GPUS 逗号个数推算，改上面一行即可，
# 不会出现「可见 2 张卡却起 8 个进程」这种对不上的情况）
NPROC=$(awk -F',' '{print NF}' <<< "${GPUS}")

# COCO 扁平数据集根目录（其下有 train/ 与 test/ 子目录）
DATA_ROOT="/root/autodl-tmp/wds_mscoco_captions2017"

# 本机环境的 OMP_NUM_THREADS 被设成了 0（无效值），会让 libgomp 打印
# "Invalid value for environment variable OMP_NUM_THREADS" 并回退到默认值。
# 训练时是 2 个主进程 + 16 个 dataloader worker 共用 32 核，若每个进程再各自
# 开一个 OpenMP 线程池会严重超订、上下文切换吃掉吞吐，所以固定为 1
# （数据增强本身是 PIL 的单线程操作，不依赖 BLAS 多线程）。
export OMP_NUM_THREADS=1

OUTPUT_DIR="./output"

# ---------------------------------------------------------------------------
# 启动前检查
# ---------------------------------------------------------------------------
if [[ ! -d "${DATA_ROOT}/train" ]]; then
    echo "错误：找不到训练数据目录 ${DATA_ROOT}/train" >&2
    exit 1
fi

# 内存检查：每个训练进程要装一份模型 + CUDA context，约 1.5GB 起。
# 无卡模式下这里的 cgroup 上限只有 2GiB，会在构建模型时被 SIGKILL(137)，
# 表现成「莫名其妙退出、没有任何 Python 报错」，所以提前提示。
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    MEM_MAX=$(cat /sys/fs/cgroup/memory.max)
    if [[ "${MEM_MAX}" != "max" ]]; then
        MEM_NEED=$((NPROC * 2 * 1024 * 1024 * 1024))   # 每个 rank 预留 2GiB
        if (( MEM_MAX < MEM_NEED )); then
            echo "警告：容器内存上限约 $(awk -v b="${MEM_MAX}" 'BEGIN{printf "%.1f", b/1024/1024/1024}') GiB，" \
                 "${NPROC} 个进程建议至少 $((NPROC * 2)) GiB，可能被 OOM Kill。" >&2
        fi
    fi
fi

echo "============================================================================"
echo "GPU        : ${GPUS}  (nproc_per_node=${NPROC})"
echo "data_root  : ${DATA_ROOT}"
echo "output_dir : ${OUTPUT_DIR}"
echo "超参见上方 train/main.py 打印的 Effective config（来源 configs/clip_vit_b32.yaml）"
echo "============================================================================"

# "$@" 允许在命令行追加/覆盖任意 Hydra 字段，例如 `bash run_train.sh epochs=5`
torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train/main.py \
    data_root="${DATA_ROOT}" \
    output_dir="${OUTPUT_DIR}" \
    "$@"
