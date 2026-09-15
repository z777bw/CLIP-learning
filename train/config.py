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
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf


# 数值字段类型归一化：YAML 1.1 会把无小数点的科学计数法（`5e-4`、`1e-6`）
# 解析成字符串，这里统一转回正确的数值类型。
_FLOAT_FIELDS = {"lr", "wd", "beta1", "beta2", "eps", "grad_clip_norm"}
_INT_FIELDS = {"image_size", "workers", "batch_size", "accumulate_steps",
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


def validate(args) -> None:
    """必填项校验。"""
    if not getattr(args, "data_manifest", None):
        raise ValueError(
            "缺少训练数据路径：请在配置文件里设置 data_manifest，"
            "或用命令行覆盖 `data_manifest=/path/to/train.jsonl`。"
        )


def print_config(cfg: DictConfig) -> None:
    """打印最终生效的配置（解析插值后的 YAML，便于复现实验）。"""
    print("=" * 60)
    print("Effective config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)
