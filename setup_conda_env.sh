#!/bin/bash

# 设置环境变量
ENV_NAME="yolov5_small_target"
ENV_PATH="/home/msb/pan1/conda_envs/$ENV_NAME"
CURRENT_DIR=$(pwd)

echo "开始创建conda环境: $ENV_NAME 到路径: $ENV_PATH"

# 创建conda环境并指定安装路径
conda create -p $ENV_PATH python=3.8 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_PATH

# 安装requirements.txt中的依赖
echo "安装基础依赖..."
pip install -r $CURRENT_DIR/requirements.txt

# 确保albumentations被安装
echo "安装小目标检测专用依赖: albumentations>=1.3.0"
pip install albumentations>=1.3.0

# 验证安装
echo "验证albumentations安装版本:"
pip show albumentations

echo "环境配置完成！"
echo "使用以下命令激活环境:"
echo "conda activate $ENV_PATH"

# 创建激活环境的快捷脚本
cat > $CURRENT_DIR/activate_env.sh << EOL
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_PATH
EOL

chmod +x $CURRENT_DIR/activate_env.sh
echo "或者直接运行: source ./activate_env.sh"