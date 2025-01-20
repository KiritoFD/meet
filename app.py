from config.settings import POSE_CONFIG
from pose.pipeline import PosePipeline

def process_video():
    # 使用配置创建pipeline
    pipeline = PosePipeline(config=POSE_CONFIG)
    
    try:
        # 处理视频帧
        while True:
            frame = capture.read()
            if frame is None:
                break
                
            # 使用管线处理
            result = pipeline.process_frame(frame)
            if result is not None:
                # 显示或保存结果
                cv2.imshow('result', result)
                
    except Exception as e:
        logger.error(f"处理错误: {e}")
        
    finally:
        # 清理资源
        pipeline.release() 