"""上传模型到 ModelScope"""
import os, json
from modelscope.hub.api import HubApi

# 登录（需要 token）
api = HubApi()
api.login("YOUR_MODELSCOPE_TOKEN")  # 替换为你的真实 token

# 创建 repo
repo_id = "SPA3K/quantpilot-models"
try:
    api.create_repo(repo_id, repo_type="model")
    print(f"✅ 创建 repo: {repo_id}")
except Exception as e:
    print(f"Repo: {e}")

# 上传模型
base = os.path.expanduser("~/.quantpilot/models/prebuilt")
for sector in sorted(os.listdir(base)):
    sector_dir = os.path.join(base, sector)
    if not os.path.isdir(sector_dir):
        continue
    print(f"\n📦 {sector}/")
    for fname in sorted(os.listdir(sector_dir)):
        fpath = os.path.join(sector_dir, fname)
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=fpath,
            path_in_repo=f"{sector}/{fname}",
            repo_type="model",
        )
        print(f"  ✅ {fname}")

print(f"\n🎉 https://modelscope.cn/models/{repo_id}")
