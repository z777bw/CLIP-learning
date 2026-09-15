# -*- coding: utf-8 -*-
"""CLIP 分布式训练代码包。

模块划分：
  - main  : 训练主入口（参数解析、分布式初始化、训练循环、checkpoint）
  - dist  : 分布式环境初始化与工具函数
  - data  : 数据加载（Dataset / DistributedSampler / DataLoader）
  - loss  : CLIP 对比损失（InfoNCE + all_gather local loss）
  - optim : 模型构建、优化器、学习率调度
  - utils : checkpoint 保存/加载

运行方式（在仓库根目录下执行，见根目录 run_train.sh）：
    torchrun --nproc_per_node=8 train/main.py --model ViT-B/32 ...
"""
