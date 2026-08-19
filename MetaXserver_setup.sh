#!/usr/bin/env bash

# 遇到错误立即退出，未声明变量报错
set -euo pipefail

# 颜色定义
GREEN='\033[0;32m'
NC='\033[0m' # 无颜色

echo -e "${GREEN}[1/9] Update the software package list...${NC}"
sudo apt update

echo -e "${GREEN}[2/9] Download Git...${NC}"
sudo apt install -y git

echo -e "${GREEN}[3/9 & 4/9] Configure the Git global username and email address...${NC}"
git config --global user.name "Yiran-SUSTech"
git config --global user.email "yiranzhang3502@outlook.com"

echo -e "${GREEN}[5/9] Generate ED25519 SSH key...${NC}"
mkdir -p ~/.ssh
chmod 700 ~/.ssh

KEY_PATH="$HOME/.ssh/id_ed25519_yiran"
if [ -f "$KEY_PATH" ]; then
    echo "SSH secret key $KEY_PATH 已存在，跳过生成。"
else
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N ""
fi

echo -e "${GREEN}[6/9] 配置 ~/.ssh/config...${NC}"
CONFIG_PATH="$HOME/.ssh/config"

# 如果配置尚未存在，追加配置到 ~/.ssh/config
if ! grep -q "IdentityFile ~/.ssh/id_ed25519_yiran" "$CONFIG_PATH" 2>/dev/null; then
    cat << 'EOF' >> "$CONFIG_PATH"

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_yiran
EOF
    chmod 600 "$CONFIG_PATH"
    echo "SSH 配置已写入 $CONFIG_PATH"
else
    echo "SSH 配置已存在，跳过写入。"
fi

echo -e "${GREEN}[7/9] SSH 公钥已生成，请复制下方内容并添加到 GitHub (Settings -> SSH and GPG keys):${NC}"
echo "--------------------------------------------------------------------------------"
cat "${KEY_PATH}.pub"
echo "--------------------------------------------------------------------------------"

# # 暂停并等待手动添加公钥
# read -rp "请将上述公钥添加至 GitHub 后，按 [Enter] 键继续测试连接..."

# echo -e "${GREEN}[8/9] 测试 GitHub SSH 连接...${NC}"
# # ssh -T 成功时返回码为 1，使用 || true 避免 set -e 导致脚本意外终止
# ssh -T -o StrictHostKeyChecking=accept-new git@github.com || true

echo -e "${GREEN}[9/9] 设置 Hugging Face 镜像环境变量...${NC}"
HF_LINE='export HF_ENDPOINT=https://hf-mirror.com'

# 写入当前 shell 环境
export HF_ENDPOINT=https://hf-mirror.com

# # 自动持久化到 ~/.bashrc（避免下次重新登录后失效）
# if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
#     echo "$HF_LINE" >> ~/.bashrc
#     echo "已将 HF_ENDPOINT 环境变量写入 ~/.bashrc"
# fi

echo -e "\n${GREEN}=== 所有配置已完成！ ===${NC}"