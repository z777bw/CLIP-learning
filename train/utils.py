# -*- coding: utf-8 -*-
"""训练相关工具：checkpoint 保存/加载、日志指标等。"""
import csv
import os
import re
import time

import torch


# metrics.csv 的列顺序（表头只在文件为空时写一次，断点续训追加时不会重复写）
_CSV_FIELDS = ["epoch", "step", "total_steps", "loss", "lr", "samples_per_s", "elapsed_s"]

# 匹配 save_checkpoint 产出的文件名，用于滚动清理
_CKPT_RE = re.compile(r"^checkpoint_ep(\d+)\.pt$")


def _timestamp() -> str:
    """统一的时间戳格式，便于事后按时间检索日志。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class TrainLogger:
    """训练日志：控制台交给 tqdm 进度条，本类负责落盘。

    在 ``output_dir`` 下产出两个文件：

      * ``train.log``   —— 人类可读。每次运行的完整配置、周期性 step 记录、
                           每个 epoch 的汇总、checkpoint 记录。
      * ``metrics.csv`` —— 机器可读。每个日志点一行，列固定（见表头），
                           可直接 ``pandas.read_csv`` 画 loss / lr 曲线。

    为什么要有两个文件：``train.log`` 用来「事后复盘某一步发生了什么」，
    而 ``metrics.csv`` 用来「把曲线画出来」—— 后者如果靠正则去解析前者的
    文本，格式一改就全废了。

    三个输出通道的分工（避免互相打架）：
      * 进度条（tqdm）：控制台实时刷新，显示 loss / lr / samples/s；
      * ``log_step``   ：**只写文件**。控制台已经有进度条了，再 print 会把
                         进度条冲得七零八落；
      * ``log_epoch``  ：控制台 + 文件。此时进度条已关闭，打一行汇总正合适。

    多进程：只在主进程实例化（``enabled=True``），其余 rank 传 ``enabled=False``
    让所有方法退化成空操作 —— 否则多个进程会同时往同一个文件里写，内容交错。
    注意 ``TrainLogger`` 只管自己这一份文件句柄，跨进程的 loss 汇总仍由调用方
    用 ``dist.all_reduce`` 完成（集合通信每个 rank 都必须参与）。

    文件以「追加」模式打开：断点续训如果写到同一个 ``output_dir``，不会把之前
    的记录冲掉；每次运行开头会写一条带时间戳的分隔头，用来划分run 边界。
    """

    def __init__(self, output_dir, enabled=True,
                 log_name="train.log", csv_name="metrics.csv"):
        self.enabled = enabled
        self.log_path = os.path.join(output_dir, log_name)
        self.csv_path = os.path.join(output_dir, csv_name)
        self._log_file = None
        self._csv_file = None
        self._csv_writer = None
        self._start_time = time.time()

        if not enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8")

        # 表头只在文件为空（新建 / 上次没写成）时写一次
        csv_is_new = (not os.path.exists(self.csv_path)
                      or os.path.getsize(self.csv_path) == 0)
        self._csv_file = open(self.csv_path, "a", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if csv_is_new:
            self._csv_writer.writerow(_CSV_FIELDS)
            self._csv_file.flush()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        """本次运行已过去的秒数（进程内计时，断点续训会从 0 重新开始）。"""
        return time.time() - self._start_time

    def _to_file(self, msg: str):
        if not self.enabled:
            return
        self._log_file.write(msg + "\n")
        # 立刻 flush：训练跑几小时，要能 tail -f 实时看，不能等缓冲区满
        self._log_file.flush()

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------
    def log_run_header(self, config_yaml, summary_lines, resume=None):
        """写入本次运行的分隔头 + 生效配置（只进文件，控制台由 main.py 打印）。"""
        if not self.enabled:
            return
        self._to_file("=" * 78)
        head = f"[{_timestamp()}] 开始训练"
        if resume:
            head += f"（断点续训自 {resume}）"
        self._to_file(head)
        self._to_file("-" * 78)
        for line in summary_lines:
            self._to_file(line)
        self._to_file("-" * 78)
        self._to_file("生效配置：")
        for line in config_yaml.rstrip().split("\n"):
            self._to_file(line)
        self._to_file("=" * 78)

    def log_step(self, epoch, step, total_steps, loss, lr, samples_per_s):
        """周期性 step 记录：写文件 + 追加一行 CSV（不打印到控制台）。"""
        if not self.enabled:
            return
        self._to_file(
            f"[{_timestamp()}] ep {epoch} step {step}/{total_steps} "
            f"loss={loss:.4f} lr={lr:.3e} samples/s={samples_per_s:.1f}"
        )
        self._csv_writer.writerow([
            epoch, step, total_steps,
            f"{loss:.6f}", f"{lr:.6e}",
            f"{samples_per_s:.2f}", f"{self.elapsed():.1f}",
        ])
        self._csv_file.flush()

    def log_epoch(self, epoch, avg_loss, samples_per_s, elapsed_s, lr):
        """epoch 汇总：控制台 + 文件（此时进度条已关闭，不会互相干扰）。"""
        if not self.enabled:
            return
        msg = (
            f"[{_timestamp()}] === epoch {epoch} 结束: avg_loss={avg_loss:.4f} "
            f"lr={lr:.3e} samples/s={samples_per_s:.1f} 耗时 {elapsed_s:.1f}s ==="
        )
        print(msg)
        self._to_file(msg)

    def log_message(self, msg):
        """其它事件（checkpoint 保存、训练结束等）：只写文件。"""
        self._to_file(f"[{_timestamp()}] {msg}")

    def close(self):
        if not self.enabled:
            return
        if self._log_file is not None:
            self._log_file.close()
        if self._csv_file is not None:
            self._csv_file.close()


def prune_checkpoints(output_dir, keep_last_n):
    """滚动保留最近 ``keep_last_n`` 个 ``checkpoint_ep*.pt``，其余删除。

    为什么必须做：每个 checkpoint 要存模型 + AdamW 的两个动量，ViT-B/32 下实测
    约 1.7 GiB。32 个 epoch 就是 54 GiB，而 ``output_dir`` 通常落在很小的系统盘上
    —— 实测跑到 epoch 10 就把 30 GiB 的盘写满，``torch.save`` 抛
    "PytorchStreamWriter failed writing file data" 并中断训练。

    Args:
        output_dir:   checkpoint 所在目录
        keep_last_n:  保留多少个；None / <=0 表示不清理

    Returns:
        被删除的文件名列表。
    """
    if not keep_last_n or keep_last_n <= 0:
        return []

    entries = []
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if m:
            entries.append((int(m.group(1)), name))

    # 必须按解析出来的整数排序，不能按文件名字典序 —— 字典序下
    # "checkpoint_ep10.pt" < "checkpoint_ep9.pt"，会把最新的当成最旧的删掉。
    entries.sort()

    removed = []
    for _, name in entries[:-keep_last_n]:
        try:
            os.remove(os.path.join(output_dir, name))
            removed.append(name)
        except OSError:
            # 清理失败不是致命问题，不能让训练因为删不掉旧文件而中断
            pass
    return removed


def save_checkpoint(args, model, optimizer, scaler, epoch, global_step, path):
    """保存训练 checkpoint（只在主进程调用）。

    存的是「完整训练状态」而非只有模型参数，目的是断点续训后能精确接上：

      * ``model``     —— 干净 state_dict（已剥 DDP 的 "module." 前缀），含 logit_scale
      * ``optimizer`` —— AdamW 的 m/v 动量，不存会让续训后 loss 出现明显跳变
      * ``scaler``    —— AMP 的 loss scale
      * ``epoch`` / ``global_step`` —— 续训起点；global_step 同时是 lr 的唯一来源

    注意这里**没有 scheduler 状态**，因为本仓库根本没有 scheduler 对象：
    ``optim.adjust_learning_rate`` 是 ``(global_step, args, total_steps)`` 的纯函数，
    只要 global_step 存对了，lr 就能原样重算出来。

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

    # 先写临时文件再原子替换，避免保存中途被 kill / 写满磁盘导致 checkpoint 损坏
    tmp_path = path + ".tmp"
    # 上次若在 torch.save 中途挂掉（例如磁盘写满），会留下一个半截的 .tmp：
    # 它本身不可用，还会白占将近一个 checkpoint 的空间，正是最缺空间的时候。
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"[checkpoint] saved to {path} (epoch={epoch}, step={global_step})")

    # 保存成功之后再清理旧的（顺序不能反：万一本步保存失败，
    # 老的 checkpoint 还在，不至于两头落空）
    keep = getattr(args, "keep_last_n", None)
    removed = prune_checkpoints(os.path.dirname(path) or ".", keep)
    if removed:
        print(f"[checkpoint] 清理旧 checkpoint（保留最近 {keep} 个）："
              f"{', '.join(removed)}")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """加载 checkpoint，返回 (epoch, global_step)。

    假设所有进程都能访问同一个文件（云上通常挂载共享存储 NFS）。
    各进程都会加载一份，保证参数一致后再包装 DDP。

    Args:
        path: checkpoint 文件路径
        model: 原始（未 DDP 包装的）模型
        optimizer / scaler: 可选，加载其状态以支持断点续训
    """
    # torch>=2.6 的 torch.load 默认 weights_only=True，只允许反序列化张量和
    # 基本类型；而本 checkpoint 里存了 args（SimpleNamespace），不在白名单内，
    # 会直接抛 UnpicklingError: "Weights only load failed"。实测除了 args 之外
    # 的字段都能安全加载，唯独它把整条续训路径毒死。
    # 这是本进程自己刚写出来的文件，来源可信，显式关掉。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

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
