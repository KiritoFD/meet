import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from connect.jitsi.meeting import JitsiMeeting

class TestJitsiMeeting:
    @pytest.fixture
    def meeting(self):
        return JitsiMeeting("test_room", "host_id")

    def test_add_participant(self, meeting):
        meeting.add_participant("user1")
        assert "user1" in meeting.participants

    def test_remove_participant(self, meeting):
        meeting.add_participant("user1")
        meeting.remove_participant("user1")
        assert "user1" not in meeting.participants

    def test_is_host(self, meeting):
        assert meeting.is_host("host_id")
        assert not meeting.is_host("user1") 