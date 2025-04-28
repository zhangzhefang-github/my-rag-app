"""
日志模块 - 提供统一的日志记录功能
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from utils.model_utils import ensure_dir_exists

def setup_logger(log_file='logs/rag_app.log', console_level=logging.DEBUG, file_level=logging.DEBUG):
    """
    设置应用日志
    
    Args:
        log_file: 日志文件路径
        console_level: 控制台输出级别
        file_level: 文件记录级别
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    ensure_dir_exists(log_dir)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置最低级别以捕获所有日志
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 创建文件处理器 (使用滚动文件处理器)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 设置三方库的日志级别为WARNING
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    
    return logger

def log_function_call(func):
    """
    函数调用日志装饰器
    """
    def wrapper(*args, **kwargs):
        try:
            logging.debug(f"调用函数 {func.__name__}")
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logging.error(f"函数 {func.__name__} 执行出错: {str(e)}", exc_info=True)
            raise
    return wrapper 