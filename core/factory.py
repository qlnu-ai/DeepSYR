import importlib
import os
from typing import Dict, Any, Optional, Type

from tasks.base import BaseTask


class Factory:
    """工厂类，负责创建模型、数据加载器、训练器和预测器的实例"""
    
    @staticmethod
    def create_task(task_type: str, **kwargs) -> BaseTask:
        """
        创建任务实例
        
        Args:
            task_type: 任务类型，如"text_classification"、"image_classification"、"object_detection"
            **kwargs: 传递给任务构造函数的额外参数
            
        Returns:
            BaseTask: 任务实例
        """
        task_module = importlib.import_module(f"tasks.{task_type}")
        
        # 查找继承自BaseTask的类
        task_class = None
        for name in dir(task_module):
            obj = getattr(task_module, name)
            if isinstance(obj, type) and issubclass(obj, BaseTask) and obj != BaseTask:
                task_class = obj
                break
                
        if task_class is None:
            raise ValueError(f"在tasks.{task_type}中未找到Task类")
            
        return task_class(**kwargs)
    
    @staticmethod
    def create_model(model_type: str, model_name: str, **kwargs):
        """
        创建模型实例
        
        Args:
            model_type: 模型类型，如"transformers"、"yolo"、"paddle"
            model_name: 模型名称
            **kwargs: 传递给模型构造函数的额外参数
            
        Returns:
            模型实例
        """
        try:
            # 尝试从我们自己的模块中导入
            module_path = f"models.{model_type}.{model_name}"
            module = importlib.import_module(module_path)
            model_class = getattr(module, "Model")  # 假设每个模型模块都有一个名为Model的类
            return model_class(**kwargs)
        except (ImportError, AttributeError):
            # 如果自定义模型不存在，尝试直接从原始库导入
            if model_type == "transformers":
                import transformers
                return getattr(transformers, model_name).from_pretrained(**kwargs)
            elif model_type == "paddle":
                import paddlenlp
                return getattr(paddlenlp.transformers, model_name).from_pretrained(**kwargs)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def create_adapter(adapter_type: str, component_type: str, **kwargs):
        """
        创建适配器实例
        
        Args:
            adapter_type: 适配器类型，如"transformers"、"yolo"、"paddle"
            component_type: 组件类型，如"trainer"、"data"、"predictor"
            **kwargs: 传递给适配器构造函数的额外参数
            
        Returns:
            适配器实例
        """
        module_path = f"adapters.{adapter_type}.{component_type}_adapter"
        module = importlib.import_module(module_path)
        
        # 查找Adapter后缀的类
        adapter_class = None
        for name in dir(module):
            if name.endswith("Adapter") and name != "Adapter":
                adapter_class = getattr(module, name)
                break
                
        if adapter_class is None:
            raise ValueError(f"在{module_path}中未找到适配器类")
            
        return adapter_class(**kwargs)
    
    @staticmethod
    def build_dataloader(backend: str, task_type: str, **config):
        """
        构建数据加载器
        
        Args:
            backend: 后端框架类型，如"transformers"、"yolo"、"paddle"
            task_type: 任务类型，如"text_classification"、"image_classification"、"object_detection"
            **config: 配置参数
            
        Returns:
            数据加载器实例
        """
        # 根据任务类型选择对应的数据加载器类
        if task_type == "text_classification":
            dataloader_class_name = "TextClassificationDataLoader"
        elif task_type == "image_classification":
            dataloader_class_name = "ImageClassificationDataLoader"
        elif task_type == "object_detection":
            dataloader_class_name = "ObjectDetectionDataLoader"
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        # 导入适配器模块
        module_path = f"adapters.{backend}.data_adapter"
        module = importlib.import_module(module_path)
        
        # 查找对应的数据加载器类
        dataloader_class = None
        for name in dir(module):
            if name.endswith(dataloader_class_name):
                dataloader_class = getattr(module, name)
                break
                
        if dataloader_class is None:
            raise ValueError(f"在{module_path}中未找到{dataloader_class_name}类")
            
        # 创建数据加载器实例
        return dataloader_class(model_name=config.get("model_name", ""), **config)
        
    @staticmethod
    def build_trainer(backend: str, task_type: str, model, **config):
        """构建训练器"""
        return Factory.create_adapter(backend, "trainer", task_type=task_type, model=model, **config)
        
    @staticmethod
    def build_predictor(backend: str, task_type: str, model, **config):
        """构建预测器"""
        return Factory.create_adapter(backend, "predictor", task_type=task_type, model=model, **config) 