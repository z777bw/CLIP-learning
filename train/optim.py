# -*- coding: utf-8 -*-
"""模型构建 + 优化器 + 学习率调度。

模型构建
--------
官方代码的 `clip.model.build_model` 需要一个 state_dict 才能反推出模型结构，
无法「从零」构建模型。这里直接根据每个模型的结构超参数实例化 CLIP 类
（其 __init__ 会自动调用 initialize_parameters 完成初始化），用于从头预训练。

优化器
------
原论文明确写明：使用 Adam + 解耦权重衰减（即 AdamW）：
    β1 = 0.9, β2 = 0.98, ε = 1e-6, weight_decay = 0.2
这是与 torch 默认 AdamW 的重要区别（默认 β2=0.999, ε=1e-8）。

学习率调度
----------
    - 前 2000 步线性 warmup，从 0 线性增加到 base_lr；
    - 之后按 cosine 衰减到 0（论文训练 32 个 epoch）。
"""
import math

import torch
from torch import nn

from clip.model import CLIP


# ---------------------------------------------------------------------------
# 标准模型的超参数配置（与官方开源 checkpoint 的结构完全一致）
# ---------------------------------------------------------------------------
_MODEL_CONFIGS = {
    # ResNet 系列：vision_layers 是四元组（每 stage 的 bottleneck 数）
    "RN50": dict(
        embed_dim=1024, image_resolution=224,
        vision_layers=(3, 4, 6, 3), vision_width=64, vision_patch_size=None,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
    ),
    "RN101": dict(
        embed_dim=512, image_resolution=224,
        vision_layers=(3, 4, 23, 3), vision_width=64, vision_patch_size=None,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
    ),
    # ViT 系列：vision_layers 是 transformer 的层数
    "ViT-B/32": dict(
        embed_dim=512, image_resolution=224,
        vision_layers=12, vision_width=768, vision_patch_size=32,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
    ),
    "ViT-B/16": dict(
        embed_dim=512, image_resolution=224,
        vision_layers=12, vision_width=768, vision_patch_size=16,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
    ),
    "ViT-L/14": dict(
        embed_dim=768, image_resolution=224,
        vision_layers=24, vision_width=1024, vision_patch_size=14,
        context_length=77, vocab_size=49408,
        transformer_width=768, transformer_heads=12, transformer_layers=12,
    ),
}


def build_clip_model(name: str) -> CLIP:
    """从零构建一个 CLIP 模型（随机初始化，用于预训练）。

    Args:
        name: 模型名，见 _MODEL_CONFIGS 的 key，如 "ViT-B/32"。

    注意：CLIP.__init__ 会调用 initialize_parameters()，完成 token embedding、
    位置编码、attention / MLP 等参数的初始化，因此无需再手动初始化。
    """
    if name not in _MODEL_CONFIGS:
        raise KeyError(f"未知模型 {name!r}，可选：{list(_MODEL_CONFIGS.keys())}")
    cfg = _MODEL_CONFIGS[name]
    return CLIP(**cfg)


class CLIPWrapper(nn.Module):
    """包装官方 CLIP 模型，让 forward 直接输出「归一化特征 + logit_scale」。

    为什么需要这个包装？
    --------------------
    官方 CLIP.forward 直接返回 logits_per_image / logits_per_text（只在本卡
    local batch 上计算），我们拿不到原始特征，无法做跨 GPU 的 all_gather。
    因此这里把 forward 重写为输出 encode_image / encode_text 的原始特征。

    同时注意一个 DDP 的**关键细节**：训练时前向必须走「DDP 包裹后的模型」，
    因为 DDP 会把梯度 all_reduce 的钩子注册在它自己包裹的模块参数上；如果
    我们绕过 DDP 直接调用底层模型的 encode_*，反向传播就不会触发梯度同步，
    导致各卡模型参数不一致。所以这个 wrapper 的 forward 走 DDP 是正确做法。
    """

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, images, texts):
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(texts)

        # L2 归一化，使内积等价于余弦相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.clip.logit_scale


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """创建 AdamW 优化器，并对不同参数分组施加不同的权重衰减。

    分组规则（常见最佳实践）：
      - bias 与所有归一化层（LayerNorm / BatchNorm / GroupNorm）的参数、
        以及标量参数（如 logit_scale）不施加权重衰减；
      - 其余权重矩阵施加 weight_decay。
    这样避免对 bias / norm 参数做衰减导致训练不稳定。
    """
    # 收集「不需要权重衰减」的参数名
    no_decay_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            for pname, _ in module.named_parameters():
                no_decay_names.add(f"{name}.{pname}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in no_decay_names or p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": args.wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),  # 原论文 (0.9, 0.98)
        eps=args.eps,                    # 原论文 1e-6
    )
    return optimizer


def adjust_learning_rate(optimizer, step: int, args, total_steps: int) -> float:
    """按当前「优化器步数」计算并设置学习率：warmup 线性上升 + cosine 衰减。

    Args:
        optimizer:   优化器
        step:        当前全局优化步数（从 0 开始，每走一个有效 batch 加 1）
        args:        参数命名空间，包含 lr / warmup
        total_steps: 总的优化步数（用于 cosine 进度归一化）

    Returns:
        当前步设置的学习率。
    """
    if step < args.warmup:
        # 线性 warmup：从 0 线性增长到 base_lr
        lr = args.lr * (step + 1) / max(1, args.warmup)
    else:
        # cosine 衰减：从 base_lr 平滑降到 0
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
