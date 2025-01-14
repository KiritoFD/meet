from flask import Blueprint, send_from_directory
import os

# 创建蓝图
static_bp = Blueprint('static', __name__)

# 修改静态文件目录路径
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'static')

@static_bp.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory(STATIC_DIR, filename) 