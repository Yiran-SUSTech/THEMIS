import os
# 确保在 Python 内部也激活国内镜像端点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi

# 初始化 API 工具
api = HfApi()

# 填入你的目标仓库 ID
repo_id = "RoninZYR/GroundingDINO-ONNX"
# 填入你服务器上 ONNX 模型的绝对路径
local_file_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx"

print("--> connecting to Hugging Face mirror...")
print(f"--> target repo: {repo_id}")

try:
    # 一键上传文件到仓库根目录
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo="groundingdino.onnx", # 上传到 HF 仓库后的文件名
        repo_id=repo_id,
        repo_type="model", # 明确是模型仓库
    )
    print("\nupload success")
except Exception as e:
    print(f"\nupload failed, please check the error message: {e}")
