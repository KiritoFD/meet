# src/api/__init__.py
"""
API模块
提供REST API接口
"""

from flask import Blueprint

api_bp = Blueprint("api", __name__)

from . import routes

__all__ = ['api_bp']