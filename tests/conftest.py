import sys
import os
from pathlib import Path

# 获取项目根目录的绝对路径
project_root = str(Path(__file__).parent.parent.absolute())

# 添加到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 插入到最前面，确保优先使用

print(f"Added to Python path: {project_root}") 