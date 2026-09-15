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
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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
    #    用 torch.amp.GradScaler("cuda", ...) 这种新写法：torch>=2.4 起
    #    torch.cuda.amp.GradScaler 已废弃，torch 2.6 下每次运行都会刷
    #    FutureWarning 到 stderr，把训练日志冲得很难看。
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    #    data_format 决定读 manifest 还是扁平目录（见 train/data.py 的 build_dataset）
    dataset = train_data.build_dataset(args, split="train")
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

    summary_lines = [
        f"model            : {args.model}",
        f"data_format      : {args.data_format}",
        f"data             : {args.data_root or args.data_manifest}",
        f"world_size       : {world_size}",
        f"batch/GPU        : {args.batch_size}",
        f"accumulate_steps : {args.accumulate_steps}",
        f"effective_batch  : {effective_batch}",
        f"dataset size     : {len(dataset)}",
        f"batches/epoch    : {num_batches_per_epoch}",
        f"steps/epoch      : {steps_per_epoch}",
        f"total_steps      : {total_steps}",
        f"AMP              : {use_amp}",
    ]
    if dist_utils.is_main_process():
        print("=" * 60)
        for line in summary_lines:
            print(line)
        print("=" * 60)

    # 9) 训练日志：train.log（人类可读）+ metrics.csv（可直接画曲线）
    #    只在主进程真正写文件，其余 rank 得到一个空操作的 logger
    logger = train_utils.TrainLogger(
        args.output_dir, enabled=dist_utils.is_main_process()
    )
    logger.log_run_header(
        OmegaConf.to_yaml(cfg, resolve=True), summary_lines, resume=args.resume
    )

    # 10) 训练循环
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
            logger=logger,
        )

        # 定期保存 checkpoint（仅主进程）
        # 最后一个 epoch 无条件保存：否则当 epochs 不是 save_freq 的整数倍时
        # （例 epochs=30 + save_freq=8 → 只在 8/16/24 存），训练跑完却拿不到
        # 最终模型，前面全白训。当前 32 % 8 == 0 恰好命中，但改 epochs 就会踩。
        is_final_epoch = (epoch + 1) == args.epochs
        if (dist_utils.is_main_process()
                and ((epoch + 1) % args.save_freq == 0 or is_final_epoch)):
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_ep{epoch + 1}.pt")
            # 保存前先取回原始模型（去掉 DDP 包装）
            train_utils.save_checkpoint(
                args, model, optimizer, scaler, epoch, global_step, ckpt_path
            )
            # save_checkpoint 自己会打印到控制台，这里只补一条文件记录，避免重复刷屏
            logger.log_message(
                f"[checkpoint] saved to {ckpt_path} "
                f"(epoch={epoch}, step={global_step})"
            )

    logger.log_message("Training finished.")
    logger.close()
    dist_utils.cleanup()
    if dist_utils.is_main_process():
        print("Training finished.")


def train_one_epoch(
    model, loader, optimizer, scaler, device, args, epoch, global_step,
    total_steps, use_amp, rank, world_size, logger,
):
    """训练一个 epoch，返回 (该 epoch 的全局平均 loss, 更新后的 global_step)。"""
    is_main = dist_utils.is_main_process()

    # 两个累加器，职责不同，不要混用：
    #   window_*  只统计「距上次打日志以来」的窗口，用于实时展示；
    #   epoch_*   统计整个 epoch（不被打日志重置），用于 epoch 结束时的真实平均 loss。
    # 旧实现只有一个累加器且在打日志时清零，于是 epoch 末尾算出的「平均」其实
    # 只是「最后一次打日志之后的平均」，并不是整个 epoch 的。
    window_loss, window_count = 0.0, 0
    epoch_loss_sum, epoch_count = 0.0, 0

    # 吞吐统计的「窗口」：只数距上次打日志以来处理了多少样本、过了多久。
    # 不能拿全局 global_step 除以「本 epoch 已耗时」—— 分母每换一个 epoch 就归零，
    # 分子却是跨 epoch 的累计值，于是每个 epoch 开头都会飙出一个假的高吞吐
    # （实测 1980 samples/s → 161110 samples/s，这种数字会误导对瓶颈的判断）。
    window_samples = 0
    last_log_time = time.time()

    # 兜底：若循环体一次都没执行（空 loader），lr 也不会是未定义变量
    lr = optimizer.param_groups[0]["lr"]
    epoch_start = time.time()

    # 进度条只在主进程渲染（其余 rank disable=True，既不输出也不做终端控制）
    # leave=False：每个 epoch 结束后进度条自行消失，只留下下面那行 epoch 汇总，
    # 避免 32 个 epoch 的进度条在终端里堆成一片。
    bar = tqdm(
        total=len(loader),
        desc=f"epoch {epoch}",
        disable=not is_main,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
        mininterval=0.5,
    )

    for step, (images, texts) in enumerate(loader):
        # non_blocking=True：配合 pin_memory，让数据搬运与计算重叠
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        # ---- 混合精度前向 ----
        with torch.amp.autocast("cuda", enabled=use_amp):
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
        step_loss = loss.item() * args.accumulate_steps
        window_loss += step_loss
        window_count += 1
        epoch_loss_sum += step_loss
        epoch_count += 1

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
            window_samples += world_size * args.batch_size * args.accumulate_steps

            if global_step % args.log_every == 0:
                # 注意：all_reduce 是集合通信，每个 rank 都必须调用。
                # 这里绝不能写成 `if is_main and ...:` —— 那样只有 rank 0 进入
                # 通信，其余 rank 直接跳过，rank 0 会在 all_reduce 上永久阻塞
                # 直到 NCCL 超时。多卡训练里「日志相关的 hang」基本都是这个原因。
                global_avg_loss = dist_utils.all_reduce_mean(
                    torch.tensor(window_loss / max(1, window_count), device=device)
                )
                now = time.time()
                speed = window_samples / max(now - last_log_time, 1e-9)

                # log_step 只写文件/CSV；控制台交给进度条（print 会把进度条冲乱）
                logger.log_step(epoch, global_step, total_steps,
                                global_avg_loss, lr, speed)
                if is_main:
                    bar.set_postfix(loss=f"{global_avg_loss:.4f}",
                                    lr=f"{lr:.2e}",
                                    sps=f"{speed:.0f}")

                window_loss, window_count = 0.0, 0
                window_samples = 0
                last_log_time = now

        bar.update(1)

    bar.close()

    # epoch 平均 loss：先算本卡均值，再跨卡取平均。
    # 各卡 batch 数相等（DistributedSampler + drop_last 保证），所以「均值的均值」
    # 就是全局均值，不需要额外按样本数加权。
    epoch_avg_loss = dist_utils.all_reduce_mean(
        torch.tensor(epoch_loss_sum / max(1, epoch_count), device=device)
    )

    epoch_elapsed = time.time() - epoch_start
    samples_per_epoch = epoch_count * world_size * args.batch_size * args.accumulate_steps
    logger.log_epoch(epoch, epoch_avg_loss,
                     samples_per_epoch / max(epoch_elapsed, 1e-9),
                     epoch_elapsed, lr)
    return epoch_avg_loss, global_step


if __name__ == "__main__":
    main()
