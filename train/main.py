# -*- coding: utf-8 -*-
"""CLIP 分布式训练主入口。

本文件是整个训练流程的「编排层」：解析参数 → 初始化分布式环境 → 构建
模型/数据/优化器 → 混合精度训练循环 → 定期保存 checkpoint。

运行方式（推荐用 torchrun，见仓库根目录 run_train.sh）：
    torchrun --nproc_per_node=8 --nnodes=1 train/main.py

超参数由 Hydra 加载 YAML 配置文件（configs/clip_vit_b32.yaml），
命令行用 key=value 语法覆盖任意字段，例如临时改 batch：
    torchrun --nproc_per_node=8 train/main.py batch_size=128 lr=1e-4

关键训练要点（均对齐 CLIP 原论文）：
  * 对比学习 InfoNCE 损失，可学习温度 logit_scale（初值 ln(1/0.07)）；
  * AdamW（β1=0.9, β2=0.98, ε=1e-6, wd=0.2）；
  * cosine 学习率 + 2000 步 warmup；
  * 大有效 batch（论文 32768）通过「多卡 + 梯度累积」组合达到；
  * 混合精度训练（PyTorch 原生 AMP，等价于原论文的 APEX O2）。

有效 batch 计算方式：
    effective_batch = world_size * batch_size * accumulate_steps
例如 8 卡 × 256 × 16 = 32768，与论文一致。
"""
import os
import sys
import time

import hydra
import torch
import torch.nn.utils as torch_nn_utils
from omegaconf import DictConfig

# 把仓库根目录加入 sys.path：
# 用 `torchrun train/main.py` 启动时，Python 只会把脚本所在目录（train/）加入
# sys.path，而官方 `clip` 包和 `train` 包都位于仓库根目录。这里显式把根目录
# （本文件的上上级目录）插入 sys.path 最前面，保证 `import clip` 与
# `from train.xxx import ...` 在任意工作目录下都能正常解析。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train import config as train_config
from train import data as train_data
from train import dist as dist_utils
from train import loss as train_loss
from train import optim as train_optim
from train import utils as train_utils


@hydra.main(version_base=None, config_path="../configs", config_name="clip_vit_b32")
def main(cfg: DictConfig):
    # 超参数：Hydra 加载 YAML + 命令行 key=value 覆盖，见 train/config.py
    args = train_config.to_namespace(cfg)
    train_config.validate(args)

    # 1) 初始化分布式环境（读取 torchrun 注入的环境变量，绑定 GPU）
    dist_utils.init_distributed_mode(args)

    # 打印最终生效配置（仅主进程，便于复现实验）
    if dist_utils.is_main_process():
        train_config.print_config(cfg)

    device = torch.device(
        f"cuda:{args.local_rank}" if args.distributed else "cuda"
    )
    world_size = args.world_size
    rank = args.rank

    # 设置随机种子：为保证各卡模型初始化一致，用同一个 seed
    torch.manual_seed(args.seed)

    # 2) 构建模型：官方 CLIP → 包装成返回特征的 wrapper → DDP
    raw_model = train_optim.build_clip_model(args.model)
    model = train_optim.CLIPWrapper(raw_model)
    model = model.to(device)

    # 3) 构建优化器（在加载 checkpoint 之前，方便加载 optimizer 状态）
    optimizer = train_optim.create_optimizer(model, args)

    # 4) AMP：GradScaler 用于混合精度下防止梯度下溢
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5) 断点续训：加载 checkpoint（在 DDP 包装之前，避免 "module." 前缀问题）
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = train_utils.load_checkpoint(
            args.resume, model, optimizer, scaler
        )
        # 续训时从「下一个 epoch」继续
        start_epoch += 1

    # 6) 包装成 DDP（必须在加载完 checkpoint、模型 .to(device) 之后）
    if args.distributed:
        model = dist_utils.wrap_ddp(model, args.local_rank)

    # 7) 构建数据集与 DataLoader
    dataset = train_data.ImageTextDataset(
        args.data_manifest,
        transform=train_data.get_train_transform(args.image_size),
    )
    train_loader, sampler = train_data.create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
    )

    # 计算总优化步数：用于 cosine 学习率调度的归一化
    # 每个 rank 的 batch 数（drop_last 后各卡相等）
    num_batches_per_epoch = len(train_loader)
    steps_per_epoch = num_batches_per_epoch // args.accumulate_steps
    total_steps = steps_per_epoch * args.epochs
    # 梯度累积后的有效 batch（与原论文 32768 对齐）
    effective_batch = world_size * args.batch_size * args.accumulate_steps

    if dist_utils.is_main_process():
        print("=" * 60)
        print(f"model            : {args.model}")
        print(f"world_size       : {world_size}")
        print(f"batch/GPU        : {args.batch_size}")
        print(f"accumulate_steps : {args.accumulate_steps}")
        print(f"effective_batch  : {effective_batch}")
        print(f"dataset size     : {len(dataset)}")
        print(f"batches/epoch    : {num_batches_per_epoch}")
        print(f"steps/epoch      : {steps_per_epoch}")
        print(f"total_steps      : {total_steps}")
        print(f"AMP              : {use_amp}")
        print("=" * 60)

    # 8) 训练循环
    model.train()
    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 重新洗牌，保证各卡采样顺序不同且随机（分布式关键）
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            use_amp=use_amp,
            rank=rank,
            world_size=world_size,
        )

        # 定期保存 checkpoint（仅主进程）
        if dist_utils.is_main_process() and (epoch + 1) % args.save_freq == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_ep{epoch + 1}.pt")
            # 保存前先取回原始模型（去掉 DDP 包装）
            train_utils.save_checkpoint(
                args, model, optimizer, scaler, epoch, global_step, ckpt_path
            )

    dist_utils.cleanup()
    if dist_utils.is_main_process():
        print("Training finished.")


def train_one_epoch(
    model, loader, optimizer, scaler, device, args, epoch, global_step,
    total_steps, use_amp, rank, world_size,
):
    """训练一个 epoch，返回 (epoch 平均 loss, 更新后的 global_step)。"""
    running_loss = 0.0
    running_count = 0
    epoch_start = time.time()

    for step, (images, texts) in enumerate(loader):
        # non_blocking=True：配合 pin_memory，让数据搬运与计算重叠
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        # ---- 混合精度前向 ----
        with torch.cuda.amp.autocast(enabled=use_amp):
            image_features, text_features, logit_scale = model(images, texts)
            loss = train_loss.clip_loss(
                image_features,
                text_features,
                logit_scale.exp(),          # 温度 τ = exp(logit_scale)
                rank=rank,
                world_size=world_size,
                local_loss=True,
            )
            # 梯度累积：把 loss 平均到每个累积步上
            loss = loss / args.accumulate_steps

        # ---- 反向传播（scaler 处理 fp16 下溢）----
        scaler.scale(loss).backward()

        # 记录 loss（乘回 accumulate_steps 得到真实单步 loss）
        running_loss += loss.item() * args.accumulate_steps
        running_count += 1

        # ---- 每 accumulate_steps 步更新一次参数 ----
        if (step + 1) % args.accumulate_steps == 0:
            # 梯度裁剪（可选）
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch_nn_utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm
                )

            scaler.step(optimizer)
            scaler.update()
            # 清空梯度：set_to_none=True 比 zero_() 更高效
            optimizer.zero_grad(set_to_none=True)

            # 更新学习率（按全局优化步数 warmup + cosine）
            lr = train_optim.adjust_learning_rate(
                optimizer, global_step, args, total_steps
            )
            global_step += 1

            # ---- 打印日志（仅主进程，loss 跨卡求平均）----
            if dist_utils.is_main_process() and global_step % args.log_every == 0:
                avg_loss = running_loss / running_count
                # 汇总所有卡的 loss，得到全局 batch 上的真实平均 loss
                global_avg_loss = dist_utils.all_reduce_mean(
                    torch.tensor(avg_loss, device=device)
                )
                speed = (global_step * world_size * args.batch_size *
                         args.accumulate_steps) / (time.time() - epoch_start)
                print(
                    f"[ep {epoch}][step {global_step}/{total_steps}] "
                    f"loss={global_avg_loss:.4f} lr={lr:.2e} "
                    f"samples/s={speed:.1f}"
                )
                running_loss = 0.0
                running_count = 0

    epoch_avg_loss = running_loss / max(1, running_count)
    return epoch_avg_loss, global_step


if __name__ == "__main__":
    main()
