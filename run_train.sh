#!/usr/bin/env bash
# ============================================================================
# 单机多卡训练启动脚本（使用 torchrun）—— 一键启动
#
# 用法：
#   bash run_train.sh                    # 按下面的默认配置训练
#   bash run_train.sh epochs=10          # 任意 Hydra 字段都能临时覆盖
#   bash run_train.sh batch_size=128 lr=1e-4
#
# 关键说明：
#   * --nproc_per_node 必须等于 CUDA_VISIBLE_DEVICES 里的卡数（下面由 GPUS 自动
#     推算，改卡号时不用再手工同步两处，避免「可见 5 张卡却起 8 个进程」）；
#   * --nnodes          = 1（单机）
#   * torchrun 会自动为每个进程注入 RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR 等
#     环境变量，train/main.py 里无需手写 multiprocessing 样板代码。
#   * 超参数统一放在 configs/clip_vit_b32.yaml，本脚本只覆盖数据路径与下面
#     显式列出的几项。
#
# 数据格式：
#   本次使用解压后的 COCO 扁平目录（<data_root>/{train,test}/s0000000.jpg + .txt）。
#   留空 data_format 会自动推断成 coco_flat；若改用 JSONL 清单，把下面换成
#   data_manifest=/path/to/train.jsonl 并把 data_format 显式设为 manifest。
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 指定使用哪几张 GPU（索引从 0 开始）。
# 注意：2/3/5 号卡被其它进程占用，所以这里只挂了空闲的 5 张。
# ---------------------------------------------------------------------------
GPUS="0,1,4,6,7"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# nproc_per_node = 卡数（由 GPUS 逗号个数推算，改上面一行即可）
NPROC=$(awk -F',' '{print NF}' <<< "${GPUS}")

# COCO 扁平数据集根目录（其下有 train/ 与 test/ 子目录）
DATA_ROOT="/root/autodl-tmp/wds_mscoco_captions2017"

# 训练超参覆盖
#   accumulate_steps=1 ：本数据集只有 11.8 万张图，若沿用配置默认的 16，
#                        每个 epoch 只剩 92/16 = 5 个优化步，32 个 epoch 共
#                        160 步 —— 比 warmup(2000) 还少，学习率全程停在 warmup
#                        里，等于没训。详见 README 的数据规模说明。
#   warmup=200         ：总步数约 92 步/epoch × 32 epoch ≈ 2900，200 步合理。
#   workers=4          ：5 个 rank × 4 = 20 个读图进程；若 samples/s 上不去说明
#                        是数据加载瓶颈（每 epoch 要解码约 19GB JPEG）。
BATCH_SIZE=256
ACCUMULATE_STEPS=1
WARMUP=200
EPOCHS=32
WORKERS=4

OUTPUT_DIR="./output"

# ---------------------------------------------------------------------------
# 启动前检查：路径写错时立即退出，不必等到模型初始化完才报错
# ---------------------------------------------------------------------------
if [[ ! -d "${DATA_ROOT}/train" ]]; then
    echo "错误：找不到训练数据目录 ${DATA_ROOT}/train" >&2
    exit 1
fi

echo "============================================================================"
echo "GPU            : ${GPUS}  (nproc_per_node=${NPROC})"
echo "data_root      : ${DATA_ROOT}"
echo "batch/GPU      : ${BATCH_SIZE}"
echo "accumulate     : ${ACCUMULATE_STEPS}"
echo "有效 batch     : $((NPROC * BATCH_SIZE * ACCUMULATE_STEPS))"
echo "epochs         : ${EPOCHS}   warmup: ${WARMUP}   workers: ${WORKERS}"
echo "output_dir     : ${OUTPUT_DIR}"
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
    batch_size="${BATCH_SIZE}" \
    accumulate_steps="${ACCUMULATE_STEPS}" \
    warmup="${WARMUP}" \
    epochs="${EPOCHS}" \
    workers="${WORKERS}" \
    output_dir="${OUTPUT_DIR}" \
    "$@"
