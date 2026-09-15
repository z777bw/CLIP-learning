# -*- coding: utf-8 -*-
"""CLIP 对比学习损失函数（InfoNCE）。

核心思想
--------
CLIP 的训练数据是「图文对」(image, text)，二者一一匹配。对每个 batch：
  1. 用图像编码器得到图像特征，用文本编码器得到文本特征，并做 L2 归一化；
  2. 计算所有图像特征与所有文本特征之间的余弦相似度，得到一个 [N, N] 矩阵；
  3. 该矩阵的对角线是 N 个「正样本对」，其余 N^2 - N 个位置都是「负样本对」；
  4. 训练目标是让对角线相似度最大、其余最小，等价于对相似度矩阵分别沿
     「图像→文本」和「文本→图像」两个方向做交叉熵分类（标签就是 0..N-1）；
  5. 最终损失是这两个方向损失的均值，因此被称为「对称对比损失」。

损失可写为（τ 为可学习温度，即 logit_scale 的指数）：
    L = 0.5 * ( CE(image_logits, y) + CE(text_logits, y) )
    其中 image_logits = τ * I @ T^T,  text_logits = image_logits^T

分布式下的 local loss 技巧
--------------------------
原论文使用超大 batch size = 32768，因为负样本越多，对比学习的表示越好。
但单卡显存放不下这么大的 batch，解决办法是：
  - 用 all_gather 把各 GPU 上计算出的特征汇总，得到 [global_batch, dim] 的特征；
  - 在本卡上构造 [local_batch, global_batch] 的相似度矩阵（本卡的正样本行 vs
    所有卡的负样本列），从而获得全局的负样本数量。

关键细节：all_gather 得到的「其他卡的特征」没有梯度，只有自己本卡的
local_batch 行会产生梯度回传，这样既增大了负样本量，又不会造成梯度重复
回传或错误缩放。这个技巧通常被称为 "local loss"。
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_features(image_features, text_features, rank, world_size):
    """跨所有 GPU 汇总（all_gather）特征。

    Args:
        image_features: 本卡图像特征，shape [local_batch, dim]
        text_features:  本卡文本特征，shape [local_batch, dim]
        rank:           当前进程 rank
        world_size:     进程总数

    Returns:
        all_image_features: [global_batch, dim]（= local_batch * world_size）
        all_text_features:  [global_batch, dim]

    注意：all_gather 需要预先分配与源 tensor 形状一致的占位 tensor 列表，
    通信完成后每个进程都会持有「所有进程」的特征副本。
    """
    gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
    gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]

    dist.all_gather(gathered_image, image_features)
    dist.all_gather(gathered_text, text_features)

    all_image_features = torch.cat(gathered_image, dim=0)
    all_text_features = torch.cat(gathered_text, dim=0)
    return all_image_features, all_text_features


def clip_loss(
    image_features,
    text_features,
    logit_scale,
    rank=0,
    world_size=1,
    local_loss=True,
):
    """计算 CLIP 对称对比损失。

    Args:
        image_features: 本卡图像特征 [local_batch, dim]（已归一化）
        text_features:  本卡文本特征 [local_batch, dim]（已归一化）
        logit_scale:    可学习温度，是一个标量 tensor（调用方传入 exp 之后的值）
        rank:           当前进程 rank
        world_size:     进程总数
        local_loss:     是否使用 local loss（分布式下推荐 True，见模块注释）

    Returns:
        一个标量损失 tensor，可直接 backward。
    """
    local_batch_size = image_features.shape[0]
    device = image_features.device

    # 相似度矩阵计算放在 fp32 下进行：矩阵乘法本身很便宜（相对编码器而言），
    # 但温度缩放的 softmax 对数值精度敏感，用 fp32 更稳定。
    image_features = image_features.float()
    text_features = text_features.float()

    if world_size > 1 and local_loss:
        # ---- 分布式 local loss 路径（推荐）----
        # 拿到全局的文本/图像特征作为负样本列
        all_image, all_text = gather_features(
            image_features, text_features, rank, world_size
        )
        # 只保留本卡的正样本行 [local_batch, global_batch]
        logits_per_image = logit_scale * image_features @ all_text.t()
        logits_per_text = logit_scale * text_features @ all_image.t()

        # 本卡的正样本在全局矩阵中的真实位置：rank * local_batch + [0..local_batch)
        labels = rank * local_batch_size + torch.arange(
            local_batch_size, device=device
        )
    else:
        # ---- 单卡 / 全局 batch 路径 ----
        # 在单卡上直接构造 [N, N] 完整相似度矩阵
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(local_batch_size, device=device)

    # 对称损失：图像→文本 与 文本→图像 两个方向取平均
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    return (image_loss + text_loss) / 2.0
