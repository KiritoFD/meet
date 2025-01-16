import logging
import absl.logging

def setup_logger():
    # 禁用 absl 的日志输出
    absl.logging.set_verbosity(absl.logging.ERROR)
    logging.root.removeHandler(absl.logging._absl_handler)
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logger() 