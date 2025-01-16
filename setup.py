import os
import sys
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

def download_model():
    """下载预训练模型"""
    model_path = MODELS_DIR / "nafnet_smoother.pth"
    if model_path.exists():
        logger.info("预训练模型已存在")
        return True
        
    try:
        logger.info("开始下载预训练模型...")
        model_path.parent.mkdir(exist_ok=True)
        
        # 使用国内镜像
        url = "https://mirror.xyz/models/nafnet/NAFNet-SIDD-width32.pth"
        
        # 备用链接
        backup_urls = [
            "https://hub.fastgit.xyz/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width32.pth",
            "https://ghproxy.com/https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width32.pth"
        ]
        
        def try_download(url):
            try:
                response = requests.get(url, stream=True, timeout=10)
                total_size = int(response.headers.get('content-length', 0))
                
                if response.status_code != 200:
                    return False
                    
                with open(model_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        for data in response.iter_content(chunk_size=8192):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total_size)
                            print(f"\r下载进度: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
                            sys.stdout.flush()
                print()
                return True
            except Exception as e:
                logger.warning(f"从 {url} 下载失败: {e}")
                return False
        
        # 尝试主链接
        if try_download(url):
            logger.info("预训练模型下载完成")
            return True
            
        # 尝试备用链接
        for backup_url in backup_urls:
            logger.info(f"尝试备用链接: {backup_url}")
            if try_download(backup_url):
                logger.info("预训练模型下载完成")
                return True
                
        logger.error("所有下载链接都失败")
        return False
        
    except Exception as e:
        logger.error(f"下载预训练模型失败: {e}")
        return False

if __name__ == "__main__":
    if download_model():
        logger.info("模型下载完成！")
    else:
        logger.error("模型下载失败！") 