from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator


class BaseDataLoader(ABC):
    """数据加载器基类，定义了数据加载接口"""
    
    def __init__(self, **kwargs):
        """
        初始化数据加载器
        
        Args:
            **kwargs: 配置参数
        """
        self.config = kwargs
        
    @abstractmethod
    def load(self, data_path: str, **kwargs):
        """
        加载数据
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            加载的数据
        """
        pass
        
    @abstractmethod
    def preprocess(self, data, **kwargs):
        """
        预处理数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        pass
        
    @abstractmethod
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理数据的便捷方法
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        pass
        
    @abstractmethod
    def batch_iterator(self, data, batch_size: int, shuffle: bool = True) -> Iterator:
        """
        获取数据批次迭代器
        
        Args:
            data: 数据
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            数据批次迭代器
        """
        pass 