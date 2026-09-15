# -*- coding: utf-8 -*-
"""数据加载：图文对 Dataset + 分布式采样 + DataLoader。

数据格式
--------
原始论文使用 WebDataset 格式（把样本打包成 .tar 分片，配合在线数据增强管道）。
这里支持两种落地形式，由 ``build_dataset`` 按 ``data_format`` 分派：

1) ``manifest``：JSONL 清单，每行一个 JSON 对象，一个样本配一句 caption。
   这是便于学习和快速跑通的通用实现：
       {"image": "/data/images/0001.jpg", "caption": "a photo of a cat"}
       {"image": "/data/images/0002.jpg", "caption": "a dog running in a park"}
   对应 ``ImageTextDataset``。

2) ``coco_flat``：把 WebDataset 分片解压后的「扁平目录」形态，对应
   ``CocoFlatDataset``。目录结构为
       <data_root>/<split>/s0000000.jpg
       <data_root>/<split>/s0000000.txt
   其中 ``.txt`` 每行一句 caption（COCO 全套通常 5 句，实测还有少量 6/7 句），
   ``.jpg`` 与 ``.txt`` 同名配对。COCO train2017 约 11.8 万张、val2017 约
   5000 张。

如果你的数据规模很大（百万/亿级），建议把这两种实现换成 WebDataset 流式读取，
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
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import clip  # 复用官方 clip 包中的 tokenize


# CLIP 在 4 亿图文对上统计出的固定归一化参数（训练/推理必须一致）。
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# coco_flat 格式的文件后缀。
_IMAGE_SUFFIX = ".jpg"
_CAPTION_SUFFIX = ".txt"


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


def _scan_split_stems(split_dir: str):
    """列出 ``split_dir`` 下「图片与文本配对齐全」的样本 stem，按文件名排序。

    索引以图片为基准，并用「两个文件名集合取交集」来保证配对齐全：

      * 目录里的 ``nshards.txt`` 之类杂项文件没有同名 ``.jpg``，取交集时自然被
        排除。注意不能直接 ``glob("*.txt")`` 计数 —— 那样会把 ``nshards.txt``
        算成样本，得到一个假的样本数；
      * 万一目录是「解压了一半」的状态，缺少另一半的样本会被提前剔除并打印
        警告，而不是训练跑到第 N 步时才抛 ``FileNotFoundError``；
      * 文件名形如 ``s0000000``，7 位零填充保证字典序与数值序一致，所以直接
        ``sorted`` 即可（``s0000007`` 会排在 ``s0000010`` 前面）。
    """
    stems_with_image, stems_with_caption = set(), set()
    for name in os.listdir(split_dir):
        if name.endswith(_IMAGE_SUFFIX):
            stems_with_image.add(name[: -len(_IMAGE_SUFFIX)])
        elif name.endswith(_CAPTION_SUFFIX):
            stems_with_caption.add(name[: -len(_CAPTION_SUFFIX)])

    only_image = stems_with_image - stems_with_caption
    only_caption = stems_with_caption - stems_with_image
    if only_image or only_caption:
        print(
            f"[CocoFlatDataset] 警告：{split_dir} 下有 {len(only_image)} 张图片缺 "
            f"caption、{len(only_caption)} 个 caption 缺图片，已忽略这些样本"
        )

    return sorted(stems_with_image & stems_with_caption)


class CocoFlatDataset(Dataset):
    """读取 WebDataset 分片解压后的扁平目录：``<root>/<split>/<stem>.jpg`` + ``.txt``。

    每个 ``.txt`` 里有若干句 caption（COCO 全套通常 5 句，实测本数据集还有
    少量 6 句和 7 句），所以按实际句数随机抽取，不要写死 5。

    样本长度 == 图片数：每次 ``__getitem__`` 从该图的 caption 里随机取一句，
    于是同一张图在不同 epoch 会配到不同的 caption —— 相当于文本侧的随机增强，
    也是这里刻意让长度等于图片数（而不是展开成图文对）的原因。

    Args:
        root:      数据集根目录（其下有 train/ 和 test/ 子目录）
        split:     子目录名，训练用 "train"，评测用 "test"
        transform: 图像预处理变换
        tokenizer: 文本 tokenizer，默认使用官方 clip.tokenize
    """

    def __init__(self, root, split="train", transform=None, tokenizer=None):
        super().__init__()
        self.dir = os.path.join(root, split)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"找不到数据目录：{self.dir}")

        self.stems = _scan_split_stems(self.dir)
        if not self.stems:
            raise RuntimeError(
                f"{self.dir} 下没有找到配对的 <stem>.jpg 与 <stem>.txt"
            )

        self.transform = transform
        self.tokenizer = tokenizer if tokenizer is not None else clip.tokenize

        # 注意：不要把全部 caption 预读进内存。本数据集 train/*.txt 合计约 31MB，
        # 转成 Python str 后膨胀到几十 MB，而 world_size × workers 个进程各存
        # 一份就是上 GB，相比每步只多一次极小的文本读盘完全不划算。

    def __len__(self):
        return len(self.stems)

    def read_captions(self, idx):
        """返回第 idx 个样本的全部 caption（评测/检索需要用到全量）。"""
        path = os.path.join(self.dir, self.stems[idx] + _CAPTION_SUFFIX)
        # 这些 .txt 文件末尾没有换行符（wc -l 会少算一行），用 split("\n") 读全。
        # errors="replace" 防止个别脏字节中断整个训练。
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            captions = [line.strip() for line in f.read().split("\n")]
        captions = [c for c in captions if c]
        # 兜底：文件异常为空时返回空串，tokenize("") 至少能产出 [SOT, EOT]
        return captions or [""]

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir, self.stems[idx] + _IMAGE_SUFFIX)
        # with 确保长跑的 worker 进程不会泄漏文件描述符
        with Image.open(image_path) as im:
            # 数据集里有少量灰度图（实测约 1/800），必须转 RGB，否则
            # Normalize 前通道数对不上。
            image = self.transform(im.convert("RGB"))

        # 用模块级 random.choice，不要用 random.Random(seed ^ idx)：
        # DataLoader 会给每个 worker 用 base_seed + worker_id 播种 Python 的
        # random 全局状态，所以这里天然是 worker 安全的；而且 persistent_workers
        # 下 worker 的随机状态跨 epoch 持续推进，同一 idx 在不同 epoch 会取到
        # 不同 caption。若按 idx 定种，(图片, caption) 就成了 idx 的固定函数，
        # 每个 epoch 都取到同一句，随机增强的效果就没了。
        caption = random.choice(self.read_captions(idx))
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
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
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


def build_dataset(args, split="train"):
    """按 ``args.data_format`` 构建训练数据集（换数据格式只需改这一处）。

    Args:
        args:  训练配置（需含 data_format、image_size，以及对应格式的路径字段）
        split: 子目录名，仅 coco_flat 格式使用；manifest 是单文件，忽略该参数

    ``data_format`` 的具体取值由 ``train.config.validate`` 推断并写回，
    所以这里只会看到确定的字符串。
    """
    fmt = getattr(args, "data_format", None)
    transform = get_train_transform(args.image_size)

    if fmt == "manifest":
        return ImageTextDataset(args.data_manifest, transform=transform)

    if fmt == "coco_flat":
        return CocoFlatDataset(
            args.data_root, split=split, transform=transform
        )

    raise ValueError(
        f"未知的 data_format={fmt!r}，可选 'manifest' | 'coco_flat'"
    )
