#!/bin/bash

# 检查是否已安装 Conda
if ! command -v conda &> /dev/null; then
    echo "Conda未安装，正在下载安装程序..."
    
    # 根据系统下载对应版本
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
    else
        # Linux
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    fi
    
    # 安装
    bash $INSTALLER -b -p $HOME/miniconda3
    
    # 删除安装程序
    rm $INSTALLER
    
    # 初始化shell
    $HOME/miniconda3/bin/conda init
    
    echo "Conda安装完成！请重新运行此脚本"
    exit 0
fi

# 创建conda环境
conda env create -f meet.yaml

# 激活环境
conda activate meet

echo "环境安装完成！" 