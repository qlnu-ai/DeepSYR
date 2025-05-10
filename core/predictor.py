from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class BasePredictor(ABC):
    """预测器基类，定义了预测接口"""
    
    def __init__(self, model, **kwargs):
        """
        初始化预测器
        
        Args:
            model: 模型实例
            **kwargs: 其他参数
        """
        self.model = model
        self.config = kwargs
        
    @abstractmethod
    def preprocess(self, input_data, **kwargs):
        """
        预处理输入数据
        
        Args:
            input_data: 输入数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        pass
        
    @abstractmethod
    def predict(self, input_data, **kwargs):
        """
        使用模型进行预测
        
        Args:
            input_data: 输入数据
            **kwargs: 预测参数
            
        Returns:
            预测结果
        """
        pass
        
    @abstractmethod
    def postprocess(self, model_output, **kwargs):
        """
        后处理模型输出
        
        Args:
            model_output: 模型输出
            **kwargs: 后处理参数
            
        Returns:
            后处理后的结果
        """
        pass
        
    @abstractmethod
    def batch_predict(self, input_data_list: List, batch_size: int = 16, **kwargs):
        """
        批量预测
        
        Args:
            input_data_list: 输入数据列表
            batch_size: 批次大小
            **kwargs: 预测参数
            
        Returns:
            预测结果列表
        """
        pass 