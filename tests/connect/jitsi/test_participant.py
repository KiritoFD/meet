import pytest
import time
from connect.jitsi.participant import JitsiParticipant

@pytest.fixture
def participant():
    return JitsiParticipant("test_user", "Test User")

class TestJitsiParticipant:
    def test_basic_info(self, participant):
        """测试基本信息"""
        assert participant.user_id == "test_user"
        assert participant.display_name == "Test User"
        assert participant.is_active()

    def test_roles(self, participant):
        """测试角色管理"""
        participant.add_role("moderator")
        assert participant.has_role("moderator")
        
        participant.remove_role("moderator")
        assert not participant.has_role("moderator")

    def test_permissions(self, participant):
        """测试权限管理"""
        participant.grant_permission("kick_user")
        assert participant.has_permission("kick_user")
        
        participant.revoke_permission("kick_user")
        assert not participant.has_permission("kick_user")

    def test_activity_tracking(self, participant):
        """测试活动跟踪"""
        original_time = participant._stats.last_active
        time.sleep(0.1)
        participant.update_activity()
        assert participant._stats.last_active > original_time

    def test_message_stats(self, participant):
        """测试消息统计"""
        participant.record_message(100, is_sent=True)
        participant.record_message(200, is_sent=False)
        
        stats = participant.get_stats()
        assert stats['message_count'] == 2
        assert stats['data_sent'] == 100
        assert stats['data_received'] == 200

    def test_active_state(self, participant):
        """测试活动状态"""
        assert participant.is_active()
        
        participant.set_active(False)
        assert not participant.is_active()
        
        participant.set_active(True)
        assert participant.is_active()

    def test_stats_format(self, participant):
        """测试统计信息格式"""
        participant.add_role("speaker")
        participant.grant_permission("share_screen")
        
        stats = participant.get_stats()
        assert isinstance(stats, dict)
        assert all(k in stats for k in [
            'user_id', 'display_name', 'roles', 'permissions',
            'join_time', 'last_active', 'message_count',
            'data_sent', 'data_received', 'active'
        ])
        assert "speaker" in stats['roles']
        assert "share_screen" in stats['permissions']

    def test_default_display_name(self):
        """测试默认显示名称"""
        participant = JitsiParticipant("test_user")
        assert participant.display_name == "test_user"

    def test_multiple_roles(self, participant):
        """测试多角色"""
        roles = ["moderator", "speaker", "presenter"]
        for role in roles:
            participant.add_role(role)
            
        for role in roles:
            assert participant.has_role(role)
            
        participant.remove_role("speaker")
        assert not participant.has_role("speaker")
        assert participant.has_role("moderator")
        assert participant.has_role("presenter")