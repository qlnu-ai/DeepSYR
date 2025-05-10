from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

from core.factory import Factory


class BaseTask(ABC):
    """任务基类，定义了所有任务的通用接口"""
    
    def __init__(self, backend: str, model_name: str, **kwargs):
        """
        初始化任务
        
        Args:
            backend: 后端框架类型，如"transformers"、"yolo"、"paddle"
            model_name: 模型名称
            **kwargs: 其他配置参数
        """
        self.backend = backend
        self.model_name = model_name
        self.config = kwargs
        
        # 组件
        self.model = None
        self.dataloader = None
        self.trainer = None
        self.predictor = None
    
    def _build_components(self):
        """构建组件，包括数据加载器、训练器和预测器"""
        # 确定任务类型
        task_type = self.__class__.__name__.lower().replace("task", "")
        
        # 构建数据加载器
        if self.dataloader is None:
            self.dataloader = Factory.build_dataloader(
                backend=self.backend,
                task_type=task_type,
                model_name=self.model_name,
                **self.config
            )
        
        # 确保模型已准备
        if self.model is None:
            self.prepare_model()
            
        # 构建训练器
        if self.trainer is None:
            self.trainer = Factory.build_trainer(
                backend=self.backend,
                task_type=task_type,
                model=self.model,
                **self.config
            )
            
        # 构建预测器
        if self.predictor is None:
            self.predictor = Factory.build_predictor(
                backend=self.backend,
                task_type=task_type,
                model=self.model,
                model_name=self.model_name,
                **self.config
            )
    
    @abstractmethod
    def prepare_data(self, data_path: str, **kwargs):
        """
        准备数据
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        pass
        
    @abstractmethod
    def prepare_model(self, **kwargs):
        """
        准备模型
        
        Args:
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        pass
        
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
    def save(self, save_path: str, **kwargs):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            **kwargs: 其他参数
        """
        pass
        
    @abstractmethod
    def load(self, load_path: str, **kwargs):
        """
        加载模型
        
        Args:
            load_path: 加载路径
            **kwargs: 其他参数
            
        Returns:
            加载的模型
        """
        pass 