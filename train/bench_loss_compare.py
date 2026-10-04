# -*- coding: utf-8 -*-
"""CLIP softmax loss（全矩阵） vs SigLIP 分块 sigmoid loss：多卡基准。

对比三个变体：clip 全矩阵 / siglip 分块 / siglip 分块+每块重算(ckpt)。

用法（云上多卡机器，全局 batch 自动均分到各卡）:
    torchrun --nproc_per_node=8 train/bench_loss_compare.py   # 8 卡
    torchrun --nproc_per_node=4 train/bench_loss_compare.py   # 4 卡
直接 `python train/bench_loss_compare.py` 则为单卡 world_size=1。

测每卡的 loss 前向+反向耗时与峰值显存（不含编码器）。
"""
import math
import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from loss import clip_loss, siglip_loss, _AllGatherWithGrad

DIM = 512                       # 特征维度
SIZES = [4096, 16384, 32768]    # 全局 batch size（可自行增删）
CHUNK = 1024                    # siglip 分块的块大小
ITERS = 3


def siglip_loss_ckpt(image_features, text_features, logit_scale, chunk_size, rank, world):
    """siglip_loss + 每块 gradient checkpointing（反向重算换显存）。

    前向不保留各块的 logits/labels 中间量，反向时逐块重算 —— 训练峰值
    显存从 O(b*N) 降到 O(b*chunk)，代价是 loss 计算时间约 +30%。
    """
    def _chunk(img, txt_chunk, scale, row_offset, col_start, col_end):
        rows = row_offset + torch.arange(img.shape[0], device=img.device)
        cols = torch.arange(col_start, col_end, device=img.device)
        labels = torch.where(rows[:, None] == cols[None, :], 1.0, -1.0)
        return -F.logsigmoid(labels * (scale * img @ txt_chunk.t())).sum()

    img = image_features.float()
    txt = text_features.float()
    all_text = _AllGatherWithGrad.apply(txt, world, rank) if world > 1 else txt
    row_offset = rank * img.shape[0]

    total = torch.zeros((), device=img.device)
    for col_start in range(0, all_text.shape[0], chunk_size):
        col_end = min(col_start + chunk_size, all_text.shape[0])
        total = total + checkpoint(_chunk, img, all_text[col_start:col_end], logit_scale,
                                   row_offset, col_start, col_end,
                                   use_reentrant=False)
    return total / img.shape[0]


def main():
    if "RANK" in os.environ:      # torchrun 启动 → 真实多卡
        dist.init_process_group("nccl" if dist.is_nccl_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:                         # 普通 python 启动 → 单卡
        rank, world = 0, 1
    dev_id = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(dev_id)
    device = f"cuda:{dev_id}"

    if rank == 0:
        print(f"world_size={world}  GPU={torch.cuda.get_device_name(device)}  "
              f"torch={torch.__version__}  d={DIM}  chunk={CHUNK}")

    for n in SIZES:
        b = n // world
        torch.manual_seed(1000 + rank)
        img = F.normalize(torch.randn(b, DIM, device=device), dim=-1).requires_grad_(True)
        txt = F.normalize(torch.randn(b, DIM, device=device), dim=-1).requires_grad_(True)
        scale = torch.tensor(math.exp(2.65), device=device)

        for name, fn in [
            ("clip  全矩阵", lambda: clip_loss(img, txt, scale, rank, world)),
            ("siglip 分块", lambda: siglip_loss(img, txt, scale, rank, world, CHUNK)),
            ("siglip+ckpt重算", lambda: siglip_loss_ckpt(img, txt, scale, CHUNK, rank, world)),
        ]:
            fn().backward()                    # 预热
            img.grad = None
            txt.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                loss = fn()
                loss.backward()
                img.grad = None
                txt.grad = None
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3 / ITERS
            peak = (torch.cuda.max_memory_allocated() - base) / 1024 ** 2
            if rank == 0:
                print(f"N={n:>6} W={world} {name} | {ms:8.2f} ms/次(fwd+bwd) | "
                      f"每卡峰值 {peak:8.1f} MB | loss(rank0)={loss.item():.2f}")

        del img, txt
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
