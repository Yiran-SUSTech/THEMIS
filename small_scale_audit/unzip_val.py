import tarfile
import os

# 定义你的源文件和目标路径
src_tar = r"C:\Users\Ronin\Downloads\ILSVRC2012_img_val.tar"
target_dir = r"D:\THEMIS\small_scale_audit\ImageNet_val"

# 1. 如果目标文件夹不存在，则自动创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"已创建目标文件夹: {target_dir}")

# 2. 开始解压
print("正在解压 ImageNet-1k val 集，大约需要 1~2 分钟，请稍候...")
with tarfile.open(src_tar, 'r') as tar:
    tar.extractall(path=target_dir)

print(f"解压完成！50,000 张验证集图片已成功解压至: {target_dir}")