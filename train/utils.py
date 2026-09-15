# -*- coding: utf-8 -*-
"""训练相关工具：checkpoint 保存/加载、日志指标等。"""
import os

import torch


def save_checkpoint(args, model, optimizer, scaler, epoch, global_step, path):
    """保存训练 checkpoint（只在主进程调用）。

    Args:
        model:      可能是 DDP 包裹的模型，需要取其 .module 得到原始模型再存
        optimizer:  优化器
        scaler:     AMP 的 GradScaler（若未启用 AMP 则传 None）
        epoch:      当前 epoch（从 0 开始）
        global_step: 已完成的优化步数
        path:       保存路径
    """
    # 去掉 DDP 的 "module." 前缀，保存干净的 state_dict
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": args,
    }

    # 先写临时文件再原子替换，避免保存中途被 kill 导致 checkpoint 损坏
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"[checkpoint] saved to {path} (epoch={epoch}, step={global_step})")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """加载 checkpoint，返回 (epoch, global_step)。

    假设所有进程都能访问同一个文件（云上通常挂载共享存储 NFS）。
    各进程都会加载一份，保证参数一致后再包装 DDP。

    Args:
        path: checkpoint 文件路径
        model: 原始（未 DDP 包装的）模型
        optimizer / scaler: 可选，加载其状态以支持断点续训
    """
    checkpoint = torch.load(path, map_location="cpu")

    # 兼容两种保存格式：旧格式直接是 state_dict，新格式是 dict
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    print(f"[checkpoint] loaded from {path} (epoch={epoch}, step={global_step})")
    return epoch, global_step
