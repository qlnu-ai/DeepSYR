from typing import Dict, Any, List, Optional, Union
import os

from tasks.base import BaseTask
from core.factory import Factory


class ImageClassificationTask(BaseTask):
    """图像分类任务"""
    
    def __init__(self, backend: str, model_name: str, num_classes: int, **kwargs):
        """
        初始化图像分类任务
        
        Args:
            backend: 后端框架类型，目前支持"transformers"和"paddle"
            model_name: 模型名称，如"vit-base"
            num_classes: 类别数量
            **kwargs: 其他配置参数
        """
        super().__init__(backend, model_name, **kwargs)
        
        if backend not in ["transformers", "paddle"]:
            raise ValueError(f"图像分类不支持后端类型: {backend}")
            
        self.num_classes = num_classes
        
    def prepare_data(self, data_path: str, **kwargs):
        """
        准备数据
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        # 确保数据加载器已构建
        if self.dataloader is None:
            self._build_components()
        
        # 使用数据加载器处理数据
        return self.dataloader.process_data(data_path, **kwargs)
        
    def prepare_model(self, **kwargs):
        """
        准备模型
        
        Args:
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        # 合并参数
        model_kwargs = {**self.config, **kwargs, "num_labels": self.num_classes}
        
        # 创建模型
        self.model = Factory.create_model(self.backend, self.model_name, **model_kwargs)
        return self.model
        
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
        # 确保组件已构建
        self._build_components()
        
        # 使用训练器训练模型
        return self.trainer.train(train_data, eval_data, **kwargs)
        
    def evaluate(self, eval_data, **kwargs):
        """
        评估模型
        
        Args:
            eval_data: 评估数据
            **kwargs: 评估参数
            
        Returns:
            评估结果
        """
        # 确保组件已构建
        self._build_components()
        
        # 使用训练器评估模型
        return self.trainer.evaluate(eval_data, **kwargs)
        
    def predict(self, input_data, **kwargs):
        """
        使用模型进行预测
        
        Args:
            input_data: 输入数据，可以是图像路径或图像数据
            **kwargs: 预测参数
            
        Returns:
            预测结果
        """
        # 确保组件已构建
        self._build_components()
        
        # 使用预测器进行预测
        return self.predictor.predict(input_data, **kwargs)
        
    def save(self, save_path: str, **kwargs):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            **kwargs: 其他参数
        """
        if self.model is None:
            raise ValueError("模型尚未准备，请先调用prepare_model或train方法")
            
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 确保训练器已构建
        self._build_components()
        
        # 保存模型
        if self.trainer is not None:
            self.trainer.save_model(save_path, **kwargs)
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
        else:
            raise NotImplementedError(f"未实现{type(self.model)}的保存方法")
        
    def load(self, load_path: str, **kwargs):
        """
        加载模型
        
        Args:
            load_path: 加载路径
            **kwargs: 其他参数
            
        Returns:
            加载的模型
        """
        # 合并参数
        model_kwargs = {**self.config, **kwargs, "num_labels": self.num_classes}
        
        if self.backend == "transformers":
            import transformers
            self.model = transformers.AutoModelForImageClassification.from_pretrained(
                load_path, **model_kwargs
            )
        elif self.backend == "paddle":
            import paddle
            from paddle.vision.models import resnet50
            
            # 假设paddle模型使用ResNet作为基础模型
            self.model = resnet50(pretrained=False, num_classes=self.num_classes)
            state_dict = paddle.load(os.path.join(load_path, "model.pdparams"))
            self.model.set_state_dict(state_dict)
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
            
        return self.model 