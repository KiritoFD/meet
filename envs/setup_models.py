import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def setup_models():
    # 创建模型目录
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "animeganv2.pth"
    
    if not model_path.exists():
        print("正在下载预训练模型...")
        try:
            # 从 Hugging Face 下载模型
            downloaded_path = hf_hub_download(
                repo_id="Aaronboat1/animegan2-pytorch",  # 我已经上传到我的 Hugging Face 账号
                filename="face_paint_512_v2.pt",
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            
            # 重命名文件
            os.rename(downloaded_path, model_path)
            print("模型下载完成！")
            return True
            
        except Exception as e:
            print(f"下载失败: {e}")
            print("\n请通过以下方式获取模型文件:")
            print("\n方法1 - 直接下载:")
            print("1. 访问: https://huggingface.co/Aaronboat1/animegan2-pytorch/blob/main/face_paint_512_v2.pt")
            print("2. 点击 '下载' 按钮")
            print("3. 将下载的文件重命名为 'animeganv2.pth' 并放到 'models' 文件夹中")
            print("\n方法2 - 使用百度网盘:")
            print("链接: https://pan.baidu.com/s/1cG9yD7YzJdh0R3qMDM4m7g")
            print("提取码: meet")
            return False
    
    return True

if __name__ == "__main__":
    if setup_models():
        print("模型设置完成！")
    else:
        print("模型设置失败！") 