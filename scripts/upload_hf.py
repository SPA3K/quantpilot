"""上传所有模型到 HuggingFace — 在本机运行"""
from huggingface_hub import HfApi, login
import os, glob

login()  # 会提示输入 token

api = HfApi()
repo_id = "SPA3K/quantpilot-models"

# 创建 repo
api.create_repo(repo_id, repo_type="model", exist_ok=True)
print(f"✅ Repo: https://huggingface.co/{repo_id}")

# 找所有模型文件
base = os.path.expanduser("~/.quantpilot/models/prebuilt")
for sector in sorted(os.listdir(base)):
    sector_dir = os.path.join(base, sector)
    if not os.path.isdir(sector_dir):
        continue
    print(f"\n📦 {sector}/")
    for fname in sorted(os.listdir(sector_dir)):
        fpath = os.path.join(sector_dir, fname)
        path_in_repo = f"{sector}/{fname}"
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=fpath,
            path_in_repo=path_in_repo,
        )
        print(f"  ✅ {fname}")

print(f"\n🎉 全部上传完成!")
print(f"   https://huggingface.co/{repo_id}")
