import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow running test"
    )

@pytest.fixture(scope="session")
def performance_threshold():
    """性能测试阈值"""
    return {
        'max_processing_time': 0.1,  # 100ms
        'max_memory_usage': 500,     # 500MB
        'min_fps': 20,              # 20fps
        'max_latency': 50,          # 50ms
    } 