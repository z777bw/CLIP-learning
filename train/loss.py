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

SigLIP 的 sigmoid 损失（siglip_loss，分块版本）
----------------------------------------------
SigLIP (Zhai et al., 2023, https://arxiv.org/abs/2303.15343) 把上面的
「行 softmax 交叉熵」换成了「每个图文对独立的二分类 sigmoid 损失」：
    L = -1/|B| * Σ_{i,j} log σ( y_ij * x_ij )
    x_ij = τ · img_i · txt_j   （y_ij = +1 正样本对 / -1 负样本对）
softmax 的行内归一化要求整行 logit 一起参与计算；而 sigmoid loss 的每个
(i, j) 项相互独立、只需求和，因此可以把 [N, N] 的 logit 矩阵切成小块
逐块计算、逐块累加（chunked / streaming），显存从 O(N²) 降到
O(local_batch × chunk_size)，这是 SigLIP 在超大 batch 下的核心优势。
原始 CLIP 结构里没有论文中用于平衡正负样本先验的可学习偏置 b，
本实现将其忽略（等价于 b = 0）。
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


class _AllGatherWithGrad(torch.autograd.Function):
    """带梯度回传的 all_gather（torch.distributed.nn.all_gather 的简化版）。

    难点：all_gather 本身不可导 —— 本卡拿到的「其他卡的特征」是常数，
    但每张卡的 loss 都用到了整份全局特征（本卡的特征也会作为列出现在
    其他卡的 loss 里），这些梯度必须精确地汇总回特征所属的卡。

    解决：反向时每张卡手里都有「本卡 loss 对整份全局特征的梯度」。
    本卡特征的精确梯度 = 所有卡的这份梯度中「属于本卡那一段」的求和，
    即 Σ_s ∂loss_s/∂(本卡特征)。实现上对整份 grad_output 做
    all_reduce(SUM)（各卡得到相同的总和），再取出本卡对应的那一段
    （通信量等价于把 all_reduce 换成 reduce_scatter 的做法）。
    """

    @staticmethod
    def forward(ctx, tensor, world_size, rank):
        ctx.world_size = world_size
        ctx.rank = rank
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 注意不能只对本卡自己的那一段做 all_reduce：那样求和的是
        # 「各卡各自的行段」（错位的拼接），而不是「所有卡对本卡行段」
        # 的梯度。必须先汇总整份梯度，再取本卡的行段。
        grad_sum = grad_output.contiguous().clone()  # 不原地改 autograd 的缓冲
        dist.all_reduce(grad_sum)
        chunk = grad_sum.shape[0] // ctx.world_size
        grad_input = grad_sum[ctx.rank * chunk:(ctx.rank + 1) * chunk].contiguous()
        return grad_input, None, None


def siglip_loss(
    image_features,
    text_features,
    logit_scale,
    rank=0,
    world_size=1,
    chunk_size=1024,
):
    """计算 SigLIP 的成对 sigmoid 损失（分块 / streaming 版本）。

    论文: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)

    与 clip_loss 的 InfoNCE（softmax 交叉熵）不同，SigLIP 把每对 (图, 文)
    当作一个独立的二分类问题（见模块 docstring）：
        L = -1/|B| * Σ_{i,j} log σ( y_ij * x_ij )
        x_ij = logit_scale * img_i · txt_j
        y_ij = +1（i=j 正样本对）/ -1（i≠j 负样本对）
    即正样本对希望 σ(x_ij) → 1，负样本对希望 σ(-x_ij) → 1，二者地位对称，
    不需要 softmax 那样的行内归一化。

    NOTE: 论文的 logits 里还有一个可学习偏置 b（用于平衡正负样本的先验），
    本仓库是原始 CLIP 结构、没有这个参数，这里直接忽略（等价于 b = 0）。

    为什么能分块计算？
    -----------------
    softmax 交叉熵中一行的 loss 依赖整行所有 logit（归一化分母含全部负
    样本），没法只算一部分；sigmoid loss 的每个 (i, j) 项相互独立、只是
    对全体样本对求和，所以可以把 [N, N] 的 logit 矩阵沿列切成若干
    [local_batch, chunk_size] 的小块，逐块前向、逐块累加。任意时刻显存里
    只有一块 logit（O(local_batch × chunk_size) 而非 O(N²)），梯度由
    autograd 跨块自动累加。

    分布式语义
    ----------
    每张卡只负责全局矩阵中「本卡图像行 × 全局文本列」这一条带：
      1. all_gather 文本特征（带梯度回传，见 _AllGatherWithGrad），
         图像特征留在本卡，不需要通信；
      2. 沿全局文本维度分块，逐块累加条带内所有 (i, j) 的二分类 loss；
      3. 除以 local_batch_size（论文 Algorithm 1 的 1/n）。
    每个图文对 (i, j) 全局只被计算一次（在图像 i 所在的卡上），各卡 loss
    经 DDP 平均后严格等于论文的全局公式 -1/|B| ΣΣ。注意各卡的
    local_batch_size 必须相同（all_gather 的要求）。

    Args:
        image_features: 本卡图像特征 [local_batch, dim]（已归一化）
        text_features:  本卡文本特征 [local_batch, dim]（已归一化）
        logit_scale:    可学习温度，标量 tensor（调用方传入 exp 之后的值）
        rank:           当前进程 rank
        world_size:     进程总数
        chunk_size:     每块沿全局文本维度的列数；越小越省显存、循环越多

    Returns:
        一个标量损失 tensor，可直接 backward。
    """
    local_batch_size = image_features.shape[0]
    device = image_features.device

    # 与 clip_loss 相同：logits / sigmoid 的计算放在 fp32 下，数值更稳
    image_features = image_features.float()
    text_features = text_features.float()

    if world_size > 1:
        # 只需要 gather 文本：每张卡负责自己的图像行，图像特征不出本卡
        all_text = _AllGatherWithGrad.apply(text_features, world_size, rank)
    else:
        all_text = text_features

    global_batch_size = all_text.shape[0]

    # 本卡图像在全局矩阵中的行号：rank * local_batch + [0, local_batch)
    row_offset = rank * local_batch_size
    local_rows = torch.arange(local_batch_size, device=device)

    # 逐块累加本卡条带 [local_batch, global_batch] 内的 loss 之和
    total = torch.zeros((), device=device, dtype=torch.float32)
    for col_start in range(0, global_batch_size, chunk_size):
        col_end = min(col_start + chunk_size, global_batch_size)
        text_chunk = all_text[col_start:col_end]

        # 本块的 logit：τ * img_i · txt_j，shape [local_batch, chunk]
        logits = logit_scale * image_features @ text_chunk.t()

        # labels = 2 * eye - 1 的分块版本：
        # 全局列号 == 本卡图像的全局行号 → 正样本 +1，其余 → 负样本 -1
        cols = torch.arange(col_start, col_end, device=device)
        is_positive = (row_offset + local_rows).unsqueeze(1) == cols.unsqueeze(0)
        labels = torch.where(is_positive, 1.0, -1.0)

        # 累加本块所有 (i, j) 的二分类 loss：-log σ(y * x)
        total = total + (-F.logsigmoid(labels * logits)).sum()

    # 论文 Algorithm 1: l = -sum(log_sigmoid(labels * logits)) / n。
    # 这里的 sum 只是本卡条带的部分和，除以 local_batch_size 后，经 DDP
    # 对各卡取平均（×1/world_size）恰好还原为全局的 -sum / global_batch。
    return total / local_batch_size
