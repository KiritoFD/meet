# OpenPose 安装指南

## 系统要求

- Windows 10/11 或 Ubuntu 18.04+
- Python 3.7+
- CMake 3.16+
- CUDA 12.6
- Visual Studio 2022 Community

## 自动安装

1. 运行安装脚本：
```bash
python scripts/install_openpose.py
```

## 手动安装

### Windows

1. 安装依赖：
```bash
pip install cmake numpy opencv-python
```

2. 克隆 OpenPose：
```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

3. 编译：
```bash
cd openpose
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_PYTHON=ON -DDOWNLOAD_BODY_25_MODEL=ON -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
```

### Linux

1. 安装依赖：
```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev python3-dev python3-pip
```

2. 克隆 OpenPose：
```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

3. 编译：
```bash
cd openpose
mkdir build
cd build
cmake .. -DBUILD_PYTHON=ON -DDOWNLOAD_BODY_25_MODEL=ON
make -j`nproc`
```

## 验证安装

运行测试脚本：
```bash
python tests/test_openpose.py
```

## 常见问题

1. CUDA 未找到
   - 确保已安装 NVIDIA 驱动和 CUDA
   - 设置正确的环境变量

2. 编译错误
   - 确保 CMake 版本正确
   - Windows 需要 Visual Studio
   - 检查系统内存是否充足

3. Python 绑定问题
   - 确保 Python 版本兼容
   - 检查 PYTHONPATH 设置 