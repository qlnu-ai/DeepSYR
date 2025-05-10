import os
import sys
import logging
from typing import Optional, List, Union


class Logger:
    """日志管理器，提供统一的日志配置和接口"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, name: str = "DeepSYR", level: int = logging.INFO, log_dir: Optional[str] = None):
        """
        初始化日志管理器
        
        Args:
            name: 日志名称
            level: 日志级别
            log_dir: 日志目录，如果提供则会同时写入文件
        """
        if self._initialized:
            return
            
        self.name = name
        self.level = level
        self.log_dir = log_dir
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # 添加控制台处理器
        self.logger.addHandler(console_handler)
        
        # 如果提供了日志目录，添加文件处理器
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        self._initialized = True
        
    def get_logger(self):
        """获取日志器"""
        return self.logger
    
    def info(self, msg: str, *args, **kwargs):
        """记录信息日志"""
        self.logger.info(msg, *args, **kwargs)
        
    def debug(self, msg: str, *args, **kwargs):
        """记录调试日志"""
        self.logger.debug(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """记录警告日志"""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """记录错误日志"""
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(msg, *args, **kwargs)
        
    def set_level(self, level: int):
        """设置日志级别"""
        self.level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# 创建默认日志器
logger = Logger().get_logger()

# 导出便捷函数
info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
critical = logger.critical


def setup_logger(name: str = "DeepSYR", level: int = logging.INFO, log_dir: Optional[str] = None):
    """
    设置日志器
    
    Args:
        name: 日志名称
        level: 日志级别
        log_dir: 日志目录
        
    Returns:
        Logger: 日志管理器
    """
    return Logger(name, level, log_dir) 