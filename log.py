import logging
import os

def setup_logger(log_folder):
    """
    配置日志记录器，记录训练过程的信息。
    
    Args:
        log_file (str): 日志文件路径，默认保存到log文件。
    
    Returns:
        logger: 配置好的日志记录器。
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = os.path.join(log_folder, "training.log")

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器和控制台处理器
    file_handler = logging.FileHandler(log_file)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    return logger