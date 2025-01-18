import os
import sys
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent.parent)

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 打印调试信息
print("Added to Python path:", project_root)
print("Current sys.path:", sys.path) 