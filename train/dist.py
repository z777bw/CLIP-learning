# -*- coding: utf-8 -*-
"""分布式训练环境初始化与工具函数。

本文件封装了 PyTorch 分布式数据并行 (Distributed Data Parallel, DDP) 所需的
环境初始化逻辑，是整个训练代码里「分布式」部分的基石。

建议按以下顺序理解分布式训练的几个核心概念：

1. **进程与 Rank**：分布式训练中，每个 GPU 对应一个独立进程。
   - `world_size`：参与训练的进程总数（= GPU 总数）。
   - `rank`：当前进程的全局编号，范围 [0, world_size)。
   - `local_rank`：当前进程在「本机」内的编号，用于选择使用哪张 GPU。
   - rank 0 通常被称为「主进程」，负责打印日志、保存 checkpoint。

2. **进程组 (process group)**：所有进程通过 `init_process_group` 组成一个通信组，
   之后才能进行 all_reduce / all_gather / broadcast 等集合通信操作。

3. **后端 (backend)**：GPU 上使用 NCCL（NVIDIA 集合通信库），CPU 上一般用 Gloo。

4. **启动方式**：推荐使用 `torchrun`（或 `python -m torch.distributed.launch`），
   它会自动为每个进程注入 RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
   MASTER_PORT 等环境变量。我们只需读取这些变量即可完成初始化，无需自己手动
   写 multiprocessing.spawn 的样板代码。
"""
import os

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    """判断当前进程是否处于分布式环境且已完成初始化。"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """返回参与训练的进程总数（GPU 总数）。非分布式环境下返回 1。"""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """返回当前进程的全局 rank。非分布式环境下返回 0。"""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """返回当前进程在本机内的 rank（决定用哪张 GPU）。"""
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """判断当前进程是否为主进程（rank 0）。日志与保存只在主进程进行。"""
    return get_rank() == 0


def setup_for_distributed(is_master: bool):
    """把非主进程的 print 重定向到 /dev/null，避免多进程刷屏。

    只有主进程 (rank 0) 会真正打印；其余进程的 print 全部被吞掉。
    如果需要强制打印（例如报错），可以使用 print(..., force=True)。
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(cfg):
    """初始化分布式进程组，并把运行时信息写回 ``cfg``。

    依赖的环境变量（由 torchrun 自动注入）：
        MASTER_ADDR / MASTER_PORT : 主节点地址与端口
        WORLD_SIZE                : 进程总数
        RANK                      : 当前进程全局编号
        LOCAL_RANK                : 当前进程在本机内的编号

    当检测不到 RANK / WORLD_SIZE 时（例如直接 `python train.py` 单卡调试），
    自动退化为单进程模式，方便在本地快速跑通逻辑。

    注意：会向 ``cfg`` 写入 rank / world_size / local_rank / distributed 四个
    运行时字段，因此调用前需保证 cfg 可写（``OmegaConf.set_struct(cfg, False)``，
    见 main.py）。这几个值本质是「本次运行的环境信息」而非「模型超参数」，
    写到 cfg 里只是为了和超参数走同一份传递路径，省去额外的返回值。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cfg.distributed = True

        # env:// 会自动读取 MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE
        dist.init_process_group(backend="nccl", init_method="env://") # 组网建立连接
        torch.cuda.set_device(cfg.local_rank)
        print(
            f"[init] distributed training: world_size={cfg.world_size}, "
            f"rank={cfg.rank}, local_rank={cfg.local_rank}"
        )
    else:
        cfg.rank = 0
        cfg.world_size = 1
        cfg.local_rank = 0
        cfg.distributed = False
        print("[init] not using distributed mode (single process)")
        return

    # 非主进程静默
    setup_for_distributed(cfg.rank == 0)


def wrap_ddp(model: torch.nn.Module, device: int) -> torch.nn.Module:
    """把模型包装成 DistributedDataParallel，并先转换为 SyncBatchNorm。

    两个关键点：

    1. **SyncBatchNorm**：官方 CLIP 的 ResNet 变体 (RN50 等) 大量使用 BatchNorm。
       分布式训练时每个 GPU 只拿到一小块数据，单卡上的 BN 统计量噪声很大。
       SyncBatchNorm 会在 forward 时把所有卡上的 BN 统计量做一次 all_reduce，
       得到全局统计量，这在大 batch 训练中至关重要。对 ViT 变体（用 LayerNorm）
       这一步不会改变任何东西，调用它是安全的。

    2. **DDP**：把模型复制到每个 GPU 上，反向传播时自动对梯度做 all_reduce，
       使所有卡的模型参数保持一致。它是目前最常用的数据并行方案。
    """
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        # 若模型存在「某些 forward 分支没被用到」的参数，需要设为 True（更慢）。
        # 我们的训练前向会用到所有参数，故设为 False 以提升性能。
        find_unused_parameters=False,
    )
    return model


def all_reduce_mean(tensor: torch.Tensor) -> float:
    """对「标量」tensor 跨所有进程做 all_reduce 并求平均。

    用途：每张卡的 loss 只反映自己的 local batch，把所有卡的 loss 平均起来
    才是整个 global batch 上的真实 loss，用于准确记录日志。
    """
    if not is_dist_available_and_initialized():
        return tensor.item()

    world_size = get_world_size()
    # all_reduce 是就地操作，先 clone 一份避免污染原 tensor 的计算图
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / world_size
    return t.item()


def cleanup():
    """销毁进程组，释放分布式资源（训练结束时调用）。"""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()
