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

本文件只负责对加载后的 ``DictConfig`` 做校验与 ``data_format`` 推断，
训练代码直接用 ``cfg.xxx`` 访问配置，不再做任何转换。
"""
import os

from omegaconf import DictConfig, OmegaConf


_DATA_FORMATS = ("manifest", "coco_flat")


def resolve_data_format(cfg: DictConfig) -> str:
    """确定数据格式；``data_format`` 留空时按「填了哪个路径」自动推断。

    这样下面两条命令都能直接跑，不需要额外记一个开关：
        train/main.py data_manifest=/path/train.jsonl      # JSONL 清单
        train/main.py data_root=/path/to/coco_flat         # 扁平目录
    """
    if cfg.data_format:
        return cfg.data_format
    if cfg.data_root:
        return "coco_flat"
    if cfg.data_manifest:
        return "manifest"
    return "manifest"  # 两个路径都没填，交给 validate 报错


def validate(cfg: DictConfig) -> None:
    """必填项校验，并把推断出的 ``data_format`` 写回 ``cfg``。

    写回是为了让下游（``train.data.build_dataset``）只处理确定的格式字符串，
    不必重复实现一遍推断逻辑。因此调用前需保证 cfg 可写
    （``OmegaConf.set_struct(cfg, False)``，见 main.py）。
    """
    fmt = resolve_data_format(cfg)
    if fmt not in _DATA_FORMATS:
        raise ValueError(f"未知的 data_format={fmt!r}，可选：{_DATA_FORMATS}")

    if fmt == "manifest":
        if not cfg.data_manifest:
            raise ValueError(
                "缺少训练数据路径：请在配置文件里设置 data_manifest，"
                "或用命令行覆盖 `data_manifest=/path/to/train.jsonl`；"
                "也可改用扁平目录格式 `data_root=/path/to/dataset`。"
            )
    else:
        root = cfg.data_root
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

    cfg.data_format = fmt


def print_config(cfg: DictConfig) -> None:
    """打印最终生效的配置（解析插值后的 YAML，便于复现实验）。"""
    print("=" * 60)
    print("Effective config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)
