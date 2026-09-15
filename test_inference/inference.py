# -*- coding: utf-8 -*-
"""零样本分类推理：用训练好的 CLIP 权重给单张图片判定类别。

思路（CLIP 的 zero-shot 套路）：
    1. 把每个类别名套进 prompt 模板，得到一组文本（如 'a photo of a dog.'）；
    2. 文本编码器 → 每个类别一个文本 embedding，作为「分类器权重」；
    3. 图像编码器 → 一个图像 embedding；
    4. 算图像 embedding 与每个文本 embedding 的余弦相似度，取最大的那个即预测类别。

用法：
    python test_inference/inference.py                        # 用默认图片和 ep32 权重
    python test_inference/inference.py --image xxx.jpg
    python test_inference/inference.py --checkpoint output/checkpoint_ep24.pt
    python test_inference/inference.py --topk 5

关于温度：argmax 与相似度的绝对尺度无关（正数乘以任何正常数都不改变排序），
所以预测类别只由余弦相似度排序决定。打印出来的概率是用模型自己学到的
logit_scale 换算的，仅作置信度参考。
"""
import argparse
import ast
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

# 脚本在 test_inference/ 下，仓库根目录是它的上一级。
# 复用 train/ 里的模型构建和图像变换，保证推理与训练用的是**同一套**定义
# （尤其是模型必须按 build_clip_model → CLIPWrapper 的顺序包装，
#   否则 state_dict 的 "clip." 前缀会对不上，加载直接报错）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import clip  # noqa: E402  复用官方 tokenizer
from train import optim as train_optim  # noqa: E402
from train.data import get_eval_transform  # noqa: E402


def load_class_file(path):
    """解析 ``classes = [...]`` / ``templates = [...]`` 形式的文件。

    ``test_inference/cifar10.txt`` 虽然后缀是 .txt，内容却是 Python 字面量赋值。
    这里用 ``ast.literal_eval`` 而不是 ``exec`` / ``import``：只认字面量，
    任何函数调用、导入、表达式都会直接报错，绝不会执行文件里的任意代码。
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("classes", "templates"):
                found[name] = ast.literal_eval(node.value)

    missing = {"classes", "templates"} - set(found)
    if missing:
        raise ValueError(f"{path} 里没有找到 {sorted(missing)} 的赋值")
    if not found["classes"]:
        raise ValueError(f"{path} 的 classes 是空的")
    return found["classes"], found["templates"]


def build_model(checkpoint_path, device):
    """按训练时的同一套结构构建模型并载入权重。

    顺序很关键：先 ``build_clip_model`` 造出裸 CLIP，再包一层 ``CLIPWrapper``。
    训练保存的 state_dict 来自 wrapper，键名带 ``clip.`` 前缀，所以这里必须
    包同样的一层，否则 ``load_state_dict`` 会因为键名不匹配而失败。
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    # torch>=2.6 的 torch.load 默认 weights_only=True，而 checkpoint 里存了
    # args（SimpleNamespace），不在安全白名单内，会抛 UnpicklingError。
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw = train_optim.build_clip_model("ViT-B/32")
    model = train_optim.CLIPWrapper(raw)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    return model, ckpt.get("epoch"), ckpt.get("global_step")


@torch.no_grad()
def encode_texts(model, prompts, device, batch_size=256):
    """编码所有 prompt，返回 L2 归一化后的特征 [len(prompts), D]。

    分块编码：类别数 × 模板数可能上千条，一次性过文本编码器没必要。
    """
    feats = []
    for i in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[i : i + batch_size], truncate=True).to(device)
        feats.append(model.clip.encode_text(tokens))
    feats = torch.cat(feats, dim=0)
    return F.normalize(feats.float(), dim=-1)


@torch.no_grad()
def run(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"找不到图片：{args.image}")

    classes, templates = load_class_file(args.classes_file)
    model, ckpt_epoch, ckpt_step = build_model(args.checkpoint, device)

    print("=" * 66)
    print(f"checkpoint : {args.checkpoint}")
    if ckpt_epoch is not None:
        print(f"             epoch={ckpt_epoch} (即第 {ckpt_epoch + 1} 轮), step={ckpt_step}")
    print(f"device     : {device}")
    print(f"类别数     : {len(classes)}   模板数: {len(templates)}")
    print(f"logit_scale: exp={model.clip.logit_scale.exp().item():.4f} "
          f"(训练学出来的温度，下面概率一列由它换算)")
    print("=" * 66)

    # ---- 1) 文本侧：每个类别 → 一个 embedding ----
    # 官方做法是「同一类别的所有模板各编一遍，特征取平均后再归一化」，
    # 相当于对该类做了一次 prompt ensemble。只有一个模板时等价于直接编码。
    prompts = [t.format(c) for c in classes for t in templates]
    text_feats = encode_texts(model, prompts, device)
    text_feats = text_feats.reshape(len(classes), len(templates), -1).mean(dim=1)
    text_feats = F.normalize(text_feats, dim=-1)          # [C, D]

    # ---- 2) 图像侧 ----
    # 必须用评测变换（确定性 Resize+CenterCrop），不能用训练时那套随机裁剪。
    with Image.open(args.image) as im:
        raw_size, raw_mode = im.size, im.mode
        image = get_eval_transform(args.image_size)(im.convert("RGB"))
    image = image.unsqueeze(0).to(device)

    image_feats = model.clip.encode_image(image)
    image_feats = F.normalize(image_feats.float(), dim=-1)   # [1, D]

    # ---- 3) 相似度 → argmax ----
    # 预测类别只由余弦相似度的排序决定，与温度无关（正数缩放不改变 argmax）。
    sims = (image_feats @ text_feats.T).squeeze(0)           # [C]，余弦相似度
    order = sims.argsort(descending=True)
    top1 = order[0].item()

    # 概率只是给个直观参考。这里用**模型自己学出来的** logit_scale：它是训练
    # 校准出来的温度，能反映这个模型的真实置信度。若换成官方参考实现的固定
    # 100.0，10 个类的 softmax 会被压成一近 one-hot，第 2 名往后全打印成
    # "0.00%" —— 不是坏了，是被四舍五入掉了，反而看不出差距。
    logit_scale = model.clip.logit_scale.exp()
    probs = (sims * logit_scale).softmax(dim=-1)

    print(f"\n图片       : {args.image}  ({raw_size[0]}x{raw_size[1]}, {raw_mode})")
    print(f"\n>>> 预测类别: {classes[top1]}   "
          f"(余弦相似度 {sims[top1]:.4f}, 概率 {probs[top1] * 100:.3f}%)")

    k = min(args.topk, len(classes))
    print(f"\n全部 {len(classes)} 个类别的排名（前 {k}）：")
    print(f"  {'排名':<5}{'类别':<13}{'余弦相似度':>12}{'概率':>11}   {'':<22}")
    for r, i in enumerate(order[:k].tolist()):
        bar = "█" * max(0, int(round(sims[i].item() * 40)))
        mark = "  ← 预测" if r == 0 else ""
        print(f"  {r + 1:<5}{classes[i]:<13}{sims[i]:>12.4f}"
              f"{probs[i] * 100:>10.3f}%   {bar}{mark}")

    if args.show_all and k < len(classes):
        print("\n其余类别：")
        for i in order[k:].tolist():
            print(f"       {classes[i]:<13}{sims[i]:>12.4f}{probs[i] * 100:>10.3f}%")

    return classes[top1], sims


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="用训练好的 CLIP 对单张图片做零样本分类",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", default=os.path.join(here, "images.jpg"),
                   help="待分类图片")
    p.add_argument("--checkpoint", default=os.path.join(_REPO_ROOT, "output", "checkpoint_ep32.pt"),
                   help="训练产出的 checkpoint（默认用最后一轮的）")
    p.add_argument("--classes-file", default=os.path.join(here, "cifar10.txt"),
                   help="含 classes / templates 两个列表的文件")
    p.add_argument("--image-size", type=int, default=224, help="与训练一致")
    p.add_argument("--topk", type=int, default=10, help="打印前 K 个类别")
    p.add_argument("--show-all", action="store_true", help="打印全部类别（不只前 K）")
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
