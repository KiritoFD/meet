"""Jitsi meeting errors"""

class JitsiError(Exception):
    """Base error for Jitsi operations"""
    def __init__(self, message: str = None, code: int = None):
        self.message = message or "Unknown Jitsi error"
        self.code = code
        super().__init__(self.message)

class MeetingError(JitsiError):
    """Error related to meeting operations"""
    ERROR_CODES = {
        1001: "Meeting not found",
        1002: "Meeting already exists",
        1003: "Meeting is full",
        1004: "Meeting is closed",
        1005: "Invalid meeting state",
        1006: "Meeting operation timeout",
        1007: "Invalid meeting configuration"
    }

    def __init__(self, message: str = None, code: int = None):
        if code and not message:
            message = self.ERROR_CODES.get(code, "Unknown meeting error")
        super().__init__(message, code)

class ParticipantError(JitsiError):
    """Error related to participant operations"""
    ERROR_CODES = {
        2001: "Participant not found",
        2002: "Participant already exists",
        2003: "Invalid participant state",
        2004: "Participant connection failed",
        2005: "Participant authentication failed",
        2006: "Invalid participant role",
        2007: "Insufficient permissions"
    }

    def __init__(self, message: str = None, code: int = None):
        if code and not message:
            message = self.ERROR_CODES.get(code, "Unknown participant error")
        super().__init__(message, code)

class ConnectionError(JitsiError):
    """Error related to connection issues"""
    ERROR_CODES = {
        3001: "Connection failed",
        3002: "Connection timeout",
        3003: "Connection lost",
        3004: "Invalid connection state",
        3005: "Network error",
        3006: "Server unreachable",
        3007: "WebSocket error"
    }

    def __init__(self, message: str = None, code: int = None):
        if code and not message:
            message = self.ERROR_CODES.get(code, "Unknown connection error")
        super().__init__(message, code)

class MediaError(JitsiError):
    """Error related to media operations"""
    ERROR_CODES = {
        4001: "Media device not found",
        4002: "Media permission denied",
        4003: "Media stream failed",
        4004: "Invalid media format",
        4005: "Media encoding error",
        4006: "Media decoding error",
        4007: "Media quality issue"
    }

    def __init__(self, message: str = None, code: int = None):
        if code and not message:
            message = self.ERROR_CODES.get(code, "Unknown media error")
        super().__init__(message, code)

class ConfigError(JitsiError):
    """Error related to configuration"""
    ERROR_CODES = {
        5001: "Missing required config",
        5002: "Invalid config value",
        5003: "Config validation failed",
        5004: "Config load failed",
        5005: "Config save failed"
    }

    def __init__(self, message: str = None, code: int = None):
        if code and not message:
            message = self.ERROR_CODES.get(code, "Unknown config error")
        super().__init__(message, code) 