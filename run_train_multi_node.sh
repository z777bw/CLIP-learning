#!/usr/bin/env bash
# ============================================================================
# 多机多卡训练启动脚本（示例：2 个节点，每节点 8 卡，共 16 卡）
#
# 用法（在两个节点上分别执行，注意 NODE_RANK 和 MASTER_ADDR 不同）：
#
#   节点 0（主节点）：
#       NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash run_train_multi_node.sh
#   节点 1：
#       NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash run_train_multi_node.sh
#
# 注意：
#   * MASTER_ADDR 必须是主节点（node_rank=0）的内网 IP，两个节点要填成一样；
#   * MASTER_PORT 需要在所有节点上一致且未被占用；
#   * 数据目录需挂载共享存储（如 NFS），保证各节点读到同一份数据；
#   * 两个节点上代码路径需保持一致。
# ============================================================================
set -e

# 通过环境变量传入（见上面的用法说明）
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
MASTER_PORT=29500

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_MANIFEST="/path/to/your/train.jsonl"

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port=${MASTER_PORT} \
    train/main.py \
    batch_size=128 \
    data_manifest="${DATA_MANIFEST}"
