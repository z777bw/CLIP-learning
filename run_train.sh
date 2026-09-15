#!/usr/bin/env bash
# ============================================================================
# 单机多卡训练启动脚本（使用 torchrun）
#
# 用法：
#   bash run_train.sh
#
# 关键说明：
#   * --nproc_per_node  = 本机 GPU 数量（每个 GPU 启动一个进程）
#   * --nnodes          = 1（单机）
#   * torchrun 会自动为每个进程注入 RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR 等
#     环境变量，train/main.py 里无需手写 multiprocessing 样板代码。
#   * 超参数统一放在 configs/clip_vit_b32.yaml，命令行只覆盖 data_manifest。
#
# 有效 batch = world_size(8) * batch_size(256) * accumulate_steps(16) = 32768
# 与原论文的 batch size 完全一致。
# ============================================================================
set -e

# 指定使用哪几张 GPU（索引从 0 开始）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据 manifest 路径（JSONL 格式，每行 {"image": "...", "caption": "..."}）
DATA_MANIFEST="/path/to/your/train.jsonl"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train/main.py \
    data_manifest="${DATA_MANIFEST}"
