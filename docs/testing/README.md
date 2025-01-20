# 测试说明

## 测试配置
```ini
# pytest.ini
[pytest]
# 测试文件匹配模式
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 测试路径
testpaths = tests

# 日志级别
log_cli = true
log_cli_level = INFO

# 测试报告格式
addopts = 
    --verbose
    --capture=no
    --showlocals
```

## 测试结构
```
tests/
├── conftest.py          # 共享fixture
├── test_detector.py     # 检测器测试
├── test_drawer.py       # 绘制器测试
├── test_socket.py       # Socket测试
└── test_camera.py      # 摄像头测试
```

## 测试类型
1. 单元测试
   - 组件功能测试
   - 边界条件测试
   - 异常处理测试

2. 集成测试
   - 组件交互测试
   - 端到端流程测试
   - 性能测试

3. 性能测试
   - 帧率测试
   - 延迟测试
   - 内存使用测试

## 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_detector.py

# 运行特定测试类
pytest tests/test_detector.py::TestPoseDetector

# 运行特定测试方法
pytest tests/test_detector.py::TestPoseDetector::test_detect
```

## Fixtures
```python
# conftest.py
@pytest.fixture
def camera():
    """提供测试用摄像头实例"""
    camera = Camera(TEST_CONFIG['camera'])
    yield camera
    camera.release()

@pytest.fixture
def detector():
    """提供测试用检测器实例"""
    detector = PoseDetector()
    yield detector
    detector.release()

@pytest.fixture
def sample_frame():
    """提供测试用视频帧"""
    return cv2.imread('tests/data/sample.jpg')
```

## 测试覆盖率
```bash
# 生成覆盖率报告
pytest --cov=pose --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
``` 