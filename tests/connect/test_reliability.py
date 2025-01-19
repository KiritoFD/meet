import pytest
import time
import asyncio

class TestReliability:
    @pytest.mark.asyncio
    async def test_reconnection(self, unified_sender, mock_socket):
        """测试断线重连"""
        # 模拟断开连接
        mock_socket.connected = False
        
        # 发送数据触发重连
        success = await unified_sender.send(
            data_type='pose',
            data=generate_test_pose()
        )
        
        assert mock_socket.connect.called
        assert success
        assert mock_socket.connected

    @pytest.mark.asyncio
    async def test_data_integrity(self, unified_sender, mock_socket):
        """测试数据完整性"""
        test_data = generate_test_pose()
        
        # 记录发送的数据
        sent_data = []
        mock_socket.emit = lambda event, data: sent_data.append(data)
        
        await unified_sender.send(
            data_type='pose',
            data=test_data
        )
        
        # 验证数据完整性
        assert len(sent_data) == 1
        received = sent_data[0]
        assert len(received['landmarks']) == len(test_data['landmarks'])
        for sent, orig in zip(received['landmarks'], test_data['landmarks']):
            assert sent['x'] == orig['x']
            assert sent['y'] == orig['y']
            assert sent['z'] == orig['z'] 