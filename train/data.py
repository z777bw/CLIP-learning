# -*- coding: utf-8 -*-
"""数据加载：图文对 Dataset + 分布式采样 + DataLoader。

数据格式
--------
原始论文使用 WebDataset 格式（把样本打包成 .tar 分片，配合在线数据增强管道）。
为了便于学习和快速跑通，这里提供一个基于 manifest 文件的通用实现：

manifest 是一个 JSONL 文件，每一行一个 JSON 对象：
    {"image": "/data/images/0001.jpg", "caption": "a photo of a cat"}
    {"image": "/data/images/0002.jpg", "caption": "a dog running in a park"}

如果你的数据规模很大（百万/亿级），建议把 ImageTextDataset 替换成 WebDataset，
只改动本文件即可，其余训练逻辑完全不变。

分布式采样的关键点
------------------
`DistributedSampler` 会把数据集按 rank 均匀切分，保证：
  - 不同 GPU 之间不重复、不遗漏样本；
  - 每个 epoch 通过 `set_epoch(epoch)` 重新洗牌，保证各卡看到的顺序不同，
    避免训练陷入固定模式。

同时 DataLoader 必须设 `drop_last=True`：如果各卡最后一个 batch 样本数不一致，
all_gather 时会因 tensor 形状不同而报错甚至死锁。
"""
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import clip  # 复用官方 clip 包中的 tokenize


class ImageTextDataset(Dataset):
    """从 manifest (JSONL) 文件读取图文对。

    Args:
        manifest_path: JSONL 文件路径
        transform:     图像预处理变换（torchvision transforms）
        tokenizer:     文本 tokenizer，默认使用官方 clip.tokenize
    """

    def __init__(self, manifest_path, transform, tokenizer=None):
        super().__init__()
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer is not None else clip.tokenize

        self.entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.entries.append((obj["image"], obj["caption"]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_path, caption = self.entries[idx]
        # 统一转为 RGB（PNG 可能是 RGBA / 灰度）
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        # tokenize 返回 shape [1, context_length] 的 int tensor，取 [0] 得到一维
        text = self.tokenizer(caption)[0]
        return image, text


def get_train_transform(image_size: int) -> transforms.Compose:
    """CLIP 训练用的图像预处理。

    与官方推理时的 Resize + CenterCrop 不同，训练时使用 RandomResizedCrop
    做随机裁剪增强，scale 区间 (0.9, 1.0) 对应原论文的「随机方形裁剪」。
    归一化的均值/方差是 CLIP 在 4 亿图文对数据集上统计得到的固定值。
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def create_dataloader(
    dataset,
    batch_size,
    num_workers,
    distributed,
    rank,
    world_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
):
    """构建 DataLoader，并在分布式模式下自动挂载 DistributedSampler。

    返回 (dataloader, sampler)，其中 sampler 在非分布式时为 None。
    训练循环里每个 epoch 开头需要调用 sampler.set_epoch(epoch)。
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,   # 一共多少个进程（= 多少个 GPU）
            rank=rank,                 # 当前进程编号
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # 用了 sampler 之后，DataLoader 的 shuffle 必须关闭，否则会冲突
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # 常驻 worker 进程，避免每个 epoch 都重新 fork，显著加速数据加载
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    return loader, sampler
