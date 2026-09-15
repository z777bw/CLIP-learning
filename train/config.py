# -*- coding: utf-8 -*-
"""训练配置管理：基于 Hydra + OmegaConf。

分工
----
Hydra：
  * 加载 YAML 配置文件（由 main.py 的 ``@hydra.main`` 指定 config_name）；
  * 命令行覆盖，语法是 ``key=value``（注意没有 ``--`` 前缀），例如
    ``batch_size=128 lr=1e-4 data_manifest=/path/to/train.jsonl``；
  * 输出目录 / 日志管理（默认 outputs/ 下按时间戳组织，可用
    ``hydra.run.dir=...`` 覆盖）。

OmegaConf：
  * 配置的加载、点语法访问（``cfg.model`` / ``cfg["model"]``）、
    ``${...}`` 插值解析。

本文件封装了从 OmegaConf 的 ``DictConfig`` 到训练代码所用对象的转换与校验。
"""
import os
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf


# 数值字段类型归一化：YAML 1.1 会把无小数点的科学计数法（`5e-4`、`1e-6`）
# 解析成字符串，这里统一转回正确的数值类型。
_FLOAT_FIELDS = {"lr", "wd", "beta1", "beta2", "eps", "grad_clip_norm"}
_INT_FIELDS = {"image_size", "workers", "batch_size", "accumulate_steps",
               "keep_last_n",
               "epochs", "warmup", "save_freq", "log_every", "seed"}


def _coerce_types(d: dict) -> dict:
    """把数值字段统一转成正确类型（幂等，对已是数值的值也无害）。"""
    for key in _FLOAT_FIELDS:
        if d.get(key) is not None:
            d[key] = float(d[key])
    for key in _INT_FIELDS:
        if d.get(key) is not None:
            d[key] = int(d[key])
    return d


def to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """把 OmegaConf ``DictConfig`` 转成 ``SimpleNamespace``。

    - ``resolve=True`` 会解析 ``${...}`` 插值；
    - 排除 Hydra 的运行时配置（``hydra`` 键），避免污染训练参数；
    - 返回的 ``SimpleNamespace`` 支持 ``args.xxx`` 属性访问，且允许
      ``init_distributed_mode`` 动态添加 rank / world_size 等运行时字段。
    """
    d = OmegaConf.to_container(cfg, resolve=True) or {}
    d.pop("hydra", None)  # 去掉 Hydra 运行时配置
    _coerce_types(d)
    return SimpleNamespace(**d)


_DATA_FORMATS = ("manifest", "coco_flat")


def resolve_data_format(args) -> str:
    """确定数据格式；``data_format`` 留空时按「填了哪个路径」自动推断。

    这样下面两条命令都能直接跑，不需要额外记一个开关：
        train/main.py data_manifest=/path/train.jsonl      # JSONL 清单
        train/main.py data_root=/path/to/coco_flat         # 扁平目录
    """
    fmt = getattr(args, "data_format", None)
    if fmt:
        return fmt
    if getattr(args, "data_root", None):
        return "coco_flat"
    if getattr(args, "data_manifest", None):
        return "manifest"
    return "manifest"  # 两个路径都没填，交给 validate 报错


def validate(args) -> None:
    """必填项校验，并把推断出的 ``data_format`` 写回 ``args``。

    写回是为了让下游（``train.data.build_dataset``）只处理确定的格式字符串，
    不必重复实现一遍推断逻辑（``init_distributed_mode`` 也是同样的回填风格）。
    """
    fmt = resolve_data_format(args)
    if fmt not in _DATA_FORMATS:
        raise ValueError(f"未知的 data_format={fmt!r}，可选：{_DATA_FORMATS}")

    if fmt == "manifest":
        if not getattr(args, "data_manifest", None):
            raise ValueError(
                "缺少训练数据路径：请在配置文件里设置 data_manifest，"
                "或用命令行覆盖 `data_manifest=/path/to/train.jsonl`；"
                "也可改用扁平目录格式 `data_root=/path/to/dataset`。"
            )
    else:
        root = getattr(args, "data_root", None)
        if not root:
            raise ValueError(
                "缺少 data_root：请设置 `data_root=/path/to/dataset`"
                "（目录下应有 train/ 与 test/ 子目录），"
                "或用 `data_format=manifest` 配合 data_manifest。"
            )
        # 提前失败：validate 在 main.py 里远早于「构建模型 + 初始化 DDP」执行，
        # 路径写错时不至于白白花掉几分钟的模型初始化再报错。
        split_dir = os.path.join(root, "train")
        if not os.path.isdir(split_dir):
            raise ValueError(f"data_root 下找不到训练目录：{split_dir}")

    args.data_format = fmt


def print_config(cfg: DictConfig) -> None:
    """打印最终生效的配置（解析插值后的 YAML，便于复现实验）。"""
    print("=" * 60)
    print("Effective config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)
