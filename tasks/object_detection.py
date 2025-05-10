from typing import Dict, Any, List, Optional, Union
import os

from tasks.base import BaseTask
from core.factory import Factory


class ObjectDetectionTask(BaseTask):
    """目标检测任务"""
    
    def __init__(self, backend: str, model_name: str, num_classes: int, **kwargs):
        """
        初始化目标检测任务
        
        Args:
            backend: 后端框架类型，目前支持"transformers"和"yolo"
            model_name: 模型名称，如"yolov5s"或"detr-resnet-50"
            num_classes: 类别数量
            **kwargs: 其他配置参数
        """
        super().__init__(backend, model_name, **kwargs)
        
        if backend not in ["transformers", "yolo"]:
            raise ValueError(f"目标检测不支持后端类型: {backend}")
            
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
        model_kwargs = {**self.config, **kwargs, "num_classes": self.num_classes}
        
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
            预测结果，通常是检测框列表
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
        elif self.backend == "yolo":
            # YOLO模型的特殊保存方法
            if hasattr(self.model, "save"):
                self.model.save(save_path)
            else:
                # 尝试调用torch.save
                import torch
                torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
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
        model_kwargs = {**self.config, **kwargs, "num_classes": self.num_classes}
        
        if self.backend == "transformers":
            import transformers
            self.model = transformers.AutoModelForObjectDetection.from_pretrained(
                load_path, **model_kwargs
            )
        elif self.backend == "yolo":
            # 根据YOLO版本选择不同的加载方式
            if self.model_name.startswith("yolov5"):
                import torch
                self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=False)
                self.model.load_state_dict(torch.load(os.path.join(load_path, "model.pt")))
            elif self.model_name.startswith("yolov8"):
                from ultralytics import YOLO
                self.model = YOLO(os.path.join(load_path, "model.pt"))
            else:
                raise ValueError(f"不支持的YOLO模型: {self.model_name}")
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
            
        return self.model 