from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseTrainer(ABC):
    """训练器基类，定义了训练接口"""
    
    def __init__(self, model, **kwargs):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            **kwargs: 其他参数
        """
        self.model = model
        self.config = kwargs
        
    @abstractmethod
    def train(self, train_data, eval_data=None, **kwargs):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            eval_data: 评估数据
            **kwargs: 训练参数
            
        Returns:
            训练结果
        """
        pass
        
    @abstractmethod
    def evaluate(self, eval_data, **kwargs):
        """
        评估模型
        
        Args:
            eval_data: 评估数据
            **kwargs: 评估参数
            
        Returns:
            评估结果
        """
        pass
        
    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            **kwargs: 其他参数
        """
        pass
        
    @abstractmethod
    def load_model(self, load_path: str, **kwargs):
        """
        加载模型
        
        Args:
            load_path: 加载路径
            **kwargs: 其他参数
            
        Returns:
            加载的模型
        """
        pass 