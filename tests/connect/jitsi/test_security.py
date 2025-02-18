import pytest
from connect.jitsi.client import MeetingSecurity
from cryptography.exceptions import InvalidTag

class TestMeetingSecurity:
    @pytest.fixture
    def security(self):
        return MeetingSecurity(
            encryption_key=b'32byteslongencryptionkey1234567890!',
            signature_key=b'64byteslongsignaturekeyabcdefghijklmnopqrstuvwxyz1234567890@#$'
        )

    def test_encryption_cycle(self, security):
        """测试加密解密流程"""
        original_data = b"Sensitive pose data"
        
        encrypted = security.encrypt_payload(original_data)
        decrypted = security.decrypt_payload(encrypted)
        
        assert decrypted == original_data
        assert encrypted != original_data

    def test_tampered_data(self, security):
        """测试数据篡改检测"""
        data = b"Important meeting data"
        encrypted = security.encrypt_payload(data)
        
        # 篡改加密数据
        tampered = encrypted[:-5] + b"XXXXX"
        with pytest.raises(InvalidTag):
            security.decrypt_payload(tampered)

    def test_signature_verification(self, security):
        """测试签名验证机制"""
        data = b"Configuration data"
        signature = security.sign_data(data)
        
        assert security.verify_signature(data, signature) is True
        assert security.verify_signature(data + b"x", signature) is False

    @pytest.mark.parametrize("payload_size", [1024, 2048, 4096])
    def test_performance(self, security, payload_size, benchmark):
        """测试加密性能"""
        data = b"x" * payload_size
        benchmark(security.encrypt_payload, data)

    @pytest.mark.parametrize("invalid_key", [
        b'short',  # 密钥过短
        None,       # 空密钥
        'string_key'  # 非bytes类型
    ])
    def test_invalid_keys(self, invalid_key):
        """测试无效密钥处理"""
        with pytest.raises(ValueError):
            MeetingSecurity(
                encryption_key=invalid_key,
                signature_key=b'valid_key_1234567890abcdefghijklmnop'
            )

    def test_key_rotation(self):
        """测试密钥轮换机制"""
        old_security = MeetingSecurity(
            encryption_key=b'old_key_1234567890abcdefghijkl',
            signature_key=b'old_sig_1234567890abcdefghijklmnopqrstuv'
        )
        
        new_security = MeetingSecurity(
            encryption_key=b'new_key_1234567890abcdefghijkl',
            signature_key=b'new_sig_1234567890abcdefghijklmnopqrstuv'
        )
        
        data = b"Important config"
        encrypted = old_security.encrypt_payload(data)
        
        # 旧密钥应能解密
        assert old_security.decrypt_payload(encrypted) == data
        
        # 新密钥应无法解密旧数据
        with pytest.raises(InvalidTag):
            new_security.decrypt_payload(encrypted) 