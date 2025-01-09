from setuptools import setup, find_packages
import os
import sys
import site
from pathlib import Path
import io

def setup_cmake_path():
    """设置 CMake 环境变量"""
    if sys.platform == "win32":
        # 获取用户 site-packages 目录
        user_site = site.getusersitepackages()
        scripts_dir = str(Path(user_site).parent / "Scripts")
        
        # 将 CMake 路径添加到环境变量
        current_path = os.environ.get("PATH", "")
        if scripts_dir not in current_path:
            os.environ["PATH"] = scripts_dir + os.pathsep + current_path
            
            # 同时更新用户环境变量
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
                    path = winreg.QueryValueEx(key, "Path")[0]
                    if scripts_dir not in path:
                        new_path = path + os.pathsep + scripts_dir
                        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                print(f"CMake 路径已添加到环境变量: {scripts_dir}")
            except Exception as e:
                print(f"更新环境变量失败: {e}")

# 读取 README.md 文件，使用 UTF-8 编码
def read_file(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="meeting-saver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
        "cmake>=3.26.0",  # 添加 CMake 依赖
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A professional video meeting solution with real-time object removal",
    long_description=read_file("README.md"),  # 使用新的读取函数
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/meeting-saver",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'flake8>=3.9.0',
            'black>=21.0',
            'mypy>=0.900',
        ],
    },
    entry_points={
        'console_scripts': [
            'meeting-saver=src.cli:main',
        ],
    },
)

# 在安装时设置 CMake 路径
if __name__ == "__main__":
    setup_cmake_path() 