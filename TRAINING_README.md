# CLIP 分布式训练代码说明

本目录在 OpenAI 官方 CLIP 源码（只含推理/模型代码）的基础上，补充了一套
**严格对齐原论文**（*Learning Transferable Visual Models From Natural Language
Supervision*, Radford et al. 2021）的**分布式训练**代码，主要用于学习 PyTorch
分布式训练（DDP）的完整写法。

> 所有代码面向 Ubuntu + NVIDIA GPU 环境编写，使用 NCCL 后端；未在本地 Windows
> 上验证（也不需要）。

---

## 一、文件结构

训练相关代码统一放在 `train/` 目录下：

```
CLIP/
├── clip/                     # 官方源码（模型 + 推理），未改动
├── train/                    # ← 本次补充的训练代码
│   ├── __init__.py           # 包说明
│   ├── config.py             # 配置转换/校验（DictConfig → Namespace，基于 Hydra + OmegaConf）
│   ├── main.py               # 训练主入口（分布式初始化、训练循环、checkpoint）
│   ├── dist.py               # 分布式环境初始化（进程组、rank、DDP 包装、all_reduce）
│   ├── data.py               # 图文对 Dataset、DistributedSampler、DataLoader
│   ├── loss.py               # CLIP 对称对比损失（InfoNCE）+ 跨卡 all_gather 的 local loss
│   ├── optim.py              # 从零构建 CLIP 模型、AdamW、cosine+warmup 学习率
│   └── utils.py              # checkpoint 保存/加载
├── configs/                  # ← 训练配置文件
│   └── clip_vit_b32.yaml     # ViT-B/32 配置示例（对齐原论文）
├── run_train.sh              # 单机多卡启动脚本（torchrun）
├── run_train_multi_node.sh   # 多机多卡启动脚本
└── TRAINING_README.md        # 本文档
```

| 文件 | 作用 |
|------|------|
| `train/config.py` | 配置转换/校验（OmegaConf DictConfig → Namespace） |
| `train/main.py` | 主入口：分布式初始化、训练循环、checkpoint |
| `train/dist.py` | 分布式环境初始化（进程组、rank、DDP 包装、all_reduce） |
| `train/data.py` | 图文对 Dataset、DistributedSampler、DataLoader |
| `train/loss.py` | CLIP 对称对比损失（InfoNCE）+ 跨卡 all_gather 的 local loss |
| `train/optim.py` | 从零构建 CLIP 模型、AdamW、cosine+warmup 学习率 |
| `train/utils.py` | checkpoint 保存/加载 |
| `configs/clip_vit_b32.yaml` | 训练超参数配置示例 |
| `run_train.sh` | 单机多卡启动脚本（torchrun） |
| `run_train_multi_node.sh` | 多机多卡启动脚本 |

---

## 二、环境要求

```bash
# Ubuntu 20.04/22.04 + NVIDIA 驱动 + CUDA 11.x/12.x
pip install torch torchvision  # 安装与 CUDA 版本匹配的 PyTorch
pip install ftfy regex tqdm pillow  # 官方 clip 的依赖
pip install hydra-core  # 配置管理（Hydra + OmegaConf，会自动安装 omegaconf）
```

确认 PyTorch 能用 GPU 与 NCCL：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
```

---

## 三、数据准备

manifest 为 JSONL 格式，每行一个 JSON 对象（图像路径 + 文本描述）：

```json
{"image": "/data/images/0001.jpg", "caption": "a photo of a cat"}
{"image": "/data/images/0002.jpg", "caption": "a dog running in a park"}
```

大规模训练建议替换为 WebDataset（见 `train/data.py` 注释）。

---

## 四、运行方式

> 均在**仓库根目录**下执行（`train/main.py` 会自动把根目录加入 `sys.path`，
> 因此 `import clip` 与 `from train.xxx import ...` 都能正常解析）。

### 0. 配置与命令行覆盖（Hydra）

配置统一放在 `configs/clip_vit_b32.yaml`（带注释），由 `@hydra.main` 加载；
命令行用 **`key=value` 语法**（无 `--` 前缀）覆盖任意字段，例如：

```bash
python train/main.py batch_size=128 lr=1e-4 data_manifest=/path/to/train.jsonl
```

Hydra 支持更丰富的覆盖语法：

| 语法 | 含义 | 示例 |
|------|------|------|
| `key=value` | 覆盖字段 | `batch_size=128` |
| `+key=value` | 追加新字段 | `+extra_note=hello` |
| `~key` | 删除字段 | `~grad_clip_norm` |

配置里 `hydra.job.chdir: false` 保证 cwd 保持在仓库根目录（相对路径正确）。

### 1. 单机多卡

```bash
bash run_train.sh
```

核心等价命令：

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    train/main.py data_manifest=/path/to/train.jsonl
```

### 2. 多机多卡（2 节点 × 8 卡）

```bash
# 节点 0（主节点）
NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash run_train_multi_node.sh
# 节点 1
NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash run_train_multi_node.sh
```

多机时 world_size 变了，脚本里用命令行覆盖 `batch_size=128` 保持有效 batch 一致。

### 3. 单卡调试（不启动分布式）

```bash
python train/main.py data_manifest=/path/to/train.jsonl \
    batch_size=32 accumulate_steps=1 epochs=1 no_amp=true
```

未检测到 `RANK`/`WORLD_SIZE` 时，`train/main.py` 会自动退化为单进程模式。

### 4. 断点续训

```bash
torchrun --nproc_per_node=8 train/main.py \
    data_manifest=/path/to/train.jsonl \
    resume=./output/checkpoint_ep5.pt
```

---

## 五、与 CLIP 原论文的对应关系

| 论文设置 | 代码实现 | 位置 |
|----------|----------|------|
| 对比学习（InfoNCE）损失，图像↔文本对称 | `clip_loss`，两个方向交叉熵取平均 | `train/loss.py` |
| 可学习温度 τ，初值 ln(1/0.07) | `CLIP.logit_scale = np.log(1/0.07)` | 官方 `clip/model.py` |
| batch size = 32768 | `world_size × batch_size × accumulate_steps` | `train/main.py` |
| Adam + 解耦权重衰减（AdamW） | β1=0.9, β2=0.98, ε=1e-6, wd=0.2 | `train/optim.py` |
| cosine 学习率 + 2000 步 warmup | `adjust_learning_rate` | `train/optim.py` |
| 混合精度训练（原用 APEX O2） | PyTorch 原生 AMP（GradScaler + autocast） | `train/main.py` |
| 图像增强：随机方形裁剪 | `RandomResizedCrop(scale=(0.9,1.0))` | `train/data.py` |

---

## 六、分布式训练核心概念讲解

### 1. 进程与 Rank

每个 GPU 对应一个独立进程。`torchrun` 自动注入环境变量：

- `WORLD_SIZE`：进程总数（GPU 总数）
- `RANK`：当前进程全局编号 `[0, WORLD_SIZE)`
- `LOCAL_RANK`：当前进程在本机内的编号（用于 `cuda:set_device`）
- `MASTER_ADDR`/`MASTER_PORT`：主节点地址，进程间握手用

初始化只需一行（`train/dist.py`）：

```python
dist.init_process_group(backend="nccl", init_method="env://")
```

### 2. 数据划分：DistributedSampler

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
sampler.set_epoch(epoch)  # 每个 epoch 重新洗牌
```

保证各卡不重复、不遗漏地切分数据；`set_epoch` 让每个 epoch 的洗牌顺序不同。
`DataLoader` 必须 `drop_last=True`，否则 all_gather 时形状不一致会死锁。

### 3. 模型同步：DistributedDataParallel (DDP)

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # RN 系需要
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

- 每张卡持有完整模型副本，前向各自计算，反向时 DDP 自动对梯度做 all_reduce，
  保证参数一致。
- **关键坑**：前向必须走 DDP 包裹后的模型，否则梯度同步钩子不触发。本仓库用
  `CLIPWrapper`（见 `train/optim.py`）保证这一点。

### 4. 对比损失的分布式扩展：all_gather + local loss

原论文 batch=32768 是为了提供大量负样本。单卡放不下时，用 all_gather 把
各卡特征拼成全局特征，在本卡构造 `[local_batch, global_batch]` 相似度矩阵
（见 `train/loss.py`）：

```python
dist.all_gather(gathered_list, local_tensor)  # 各卡都拿到全部特征
labels = rank * local_batch + arange(local_batch)  # 正样本全局位置偏移
```

只有本卡的正样本行产生梯度，既增大负样本量，又不会梯度重复回传。

### 5. 大 batch 的两种手段：多卡 + 梯度累积

- **数据并行**：batch 随卡数线性放大；
- **梯度累积**：多个 micro-batch 的 loss 求和（或平均）后再 step 一次，等效
  增大 batch 而显存不变。

```python
loss = loss / accumulate_steps     # 平均
scaler.scale(loss).backward()
if (step + 1) % accumulate_steps == 0:
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

### 6. 混合精度（AMP）

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(images, texts) ...
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```

fp16 加速并省显存；GradScaler 通过动态缩放 loss 防止 fp16 梯度下溢。
原论文用 NVIDIA APEX 的 O2 级别，PyTorch 原生 AMP 等价且更易用。

### 7. 日志与 checkpoint

- 日志只在 rank 0 打印；loss 用 `all_reduce` 跨卡平均得到全局值。
- checkpoint 只在 rank 0 保存；`model.module.state_dict()` 去掉 DDP 的
  `module.` 前缀；临时文件 + `os.replace` 原子写，防止中断损坏。

---

## 七、常见坑（踩坑速查）

1. **卡在初始化不动**：`MASTER_PORT` 被占用或 `MASTER_ADDR` 不通，换端口/检查网络。
2. **all_gather 报 shape 不一致或死锁**：`drop_last` 没开，或各卡数据量不同。
3. **各卡 loss 一直不变/不收敛**：忘了 `sampler.set_epoch(epoch)`。
4. **单卡能跑、多卡 OOM**：单卡 batch 过大，调小 `--batch-size` 并增大
   `--accumulate-steps` 保持有效 batch 不变。
5. **fp16 出现 NaN**：先确认 `LayerNorm` 是否用了 fp32 版（官方 `model.py`
   已处理）；必要时降低 lr 或关 `--no-amp` 排查。
6. **ResNet 变体收敛差**：确认 `SyncBatchNorm.convert_sync_batchnorm` 已调用。

---

## 八、进一步学习建议

- 把 `--no-amp` 打开/关闭对比显存与速度，理解混合精度收益。
- 修改 `accumulate_steps` 观察有效 batch 对对比学习收敛的影响。
- 尝试替换 `train/data.py` 为 WebDataset，体验大规模数据管道。
- 阅读 `torch.distributed.nn.functional.all_gather`（带梯度版本）对比 local loss 的差异。
