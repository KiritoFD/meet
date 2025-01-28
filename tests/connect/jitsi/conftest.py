import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_jitsi_libs():
    """Mock Jitsi底层库"""
    with patch('connect.jitsi.transport.JitsiMeet') as mock_meet:
        mock_connection = AsyncMock()
        mock_conference = AsyncMock()
        mock_meet.return_value.connect.return_value = mock_connection
        mock_connection.join_conference.return_value = mock_conference
        yield 