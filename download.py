import os
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

# 国内hf镜像，autodl必备
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

splits = {'train': '**/train/*.tar', 'test': '**/test/*.tar'}
fs = HfFileSystem()

# 获取所有train下tar文件
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/clip-benchmark/wds_mscoco_captions2017/" + splits["train"])]

# 生成每个tar对应的hf下载url，每个单独pipe curl
urls = [
    f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {hf_hub_url(file.repo_id, file.path_in_repo, repo_type='dataset')}"
    for file in files
]

# cache_dir：下载的tar包会保存到此文件夹，下一次运行直接读本地缓存，不重复下载
ds = wds.WebDataset(urls, cache_dir="./wds_mscoco_cache").decode()

# ⚠️ 必须迭代才开始下载！
# 测试：取第一个样本，触发下载
sample = next(iter(ds))
print("成功读取样本，keys：", list(sample.keys()))

