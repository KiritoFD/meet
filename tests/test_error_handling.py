import pytest
import numpy as np
from connect.errors import (
    ConnectionError, 
    DataValidationError,
    PoseError,
    ResourceLimitError
)

def test_connection_error_handling(mock_socket):
    """测试连接错误处理"""
    mock_socket.connected = False
    
    with pytest.raises(ConnectionError):
        mock_socket.emit('test_event', {})

def test_data_validation_error_handling(setup_system):
    """测试数据验证错误处理"""
    sender = setup_system['sender']
    
    # 测试无效数据
    invalid_data = {'invalid': 'data'}
    with pytest.raises(DataValidationError):
        sender.send_pose_data(room="test", pose_results=invalid_data)
        
    # 测试空数据
    with pytest.raises(DataValidationError):
        sender.send_pose_data(room="test", pose_results=None)

def test_resource_limit_handling(setup_system):
    """测试资源限制错误处理"""
    sender = setup_system['sender']
    
    # 模拟队列满
    sender._queue.maxsize = 1
    sender._queue.put({'dummy': 'data'})
    
    with pytest.raises(ResourceLimitError):
        sender.send_pose_data(
            room="test",
            pose_results={'test': 'data'},
            timestamp=0
        )

def test_pose_error_handling(setup_system):
    """测试姿态处理错误处理"""
    pose = setup_system['pose']
    
    # 测试无效帧
    invalid_frame = np.zeros((100, 100))  # 错误的维度
    with pytest.raises(PoseError):
        pose.process(invalid_frame) 