import os
from typing import Dict, Any, Optional, List, Union

from transformers import Trainer, TrainingArguments
from datasets import Dataset

from core.trainer import BaseTrainer


class TransformersTrainerAdapter(BaseTrainer):
    """Transformers库的训练适配器"""
    
    def __init__(self, model, task_type: str, **kwargs):
        """
        初始化Transformers训练适配器
        
        Args:
            model: transformers模型实例
            task_type: 任务类型
            **kwargs: 其他参数
        """
        super().__init__(model, **kwargs)
        self.task_type = task_type
        self.trainer = None
        
    def _create_trainer(self, train_data, eval_data=None, **kwargs):
        """
        创建Trainer实例
        
        Args:
            train_data: 训练数据
            eval_data: 评估数据
            **kwargs: 训练参数
            
        Returns:
            Trainer实例
        """
        # 默认训练参数
        default_args = {
            "output_dir": "./results",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": "./logs",
            "logging_steps": 10,
            "save_steps": 500,
            "evaluation_strategy": "epoch"
        }
        
        # 合并默认参数和用户参数
        training_args = {**default_args, **kwargs}
        
        # 创建TrainingArguments
        args = TrainingArguments(**training_args)
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data
        )
        
        return self.trainer
        
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
        # 确保数据是Dataset类型
        if not isinstance(train_data, Dataset):
            raise TypeError("train_data必须是datasets.Dataset类型")
        if eval_data is not None and not isinstance(eval_data, Dataset):
            raise TypeError("eval_data必须是datasets.Dataset类型")
            
        # 创建Trainer
        trainer = self._create_trainer(train_data, eval_data, **kwargs)
        
        # 训练模型
        return trainer.train()
        
    def evaluate(self, eval_data, **kwargs):
        """
        评估模型
        
        Args:
            eval_data: 评估数据
            **kwargs: 评估参数
            
        Returns:
            评估结果
        """
        if self.trainer is None:
            self._create_trainer(None, eval_data, **kwargs)
            
        # 确保数据是Dataset类型
        if not isinstance(eval_data, Dataset):
            raise TypeError("eval_data必须是datasets.Dataset类型")
            
        # 评估模型
        return self.trainer.evaluate(eval_dataset=eval_data)
        
    def save_model(self, save_path: str, **kwargs):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            **kwargs: 其他参数
        """
        if self.trainer is None:
            raise ValueError("trainer尚未创建，请先调用train方法")
            
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        self.trainer.save_model(save_path)
        
    def load_model(self, load_path: str, **kwargs):
        """
        加载模型
        
        Args:
            load_path: 加载路径
            **kwargs: 其他参数
            
        Returns:
            加载的模型
        """
        
        # 根据任务类型加载相应的模型
        if self.task_type == "text_classification":
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        elif self.task_type == "image_classification":
            from transformers import AutoModelForImageClassification
            self.model = AutoModelForImageClassification.from_pretrained(load_path)
        elif self.task_type == "object_detection":
            from transformers import AutoModelForObjectDetection
            self.model = AutoModelForObjectDetection.from_pretrained(load_path)
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
            
        return self.model 