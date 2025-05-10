import os
from typing import Dict, Any, Optional, List, Iterator, Union
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoFeatureExtractor

from core.dataloader import BaseDataLoader


class TransformersDataAdapter(BaseDataLoader):
    """Transformers库的数据适配器"""
    
    def __init__(self, task_type: str, model_name: str, **kwargs):
        """
        初始化Transformers数据适配器
        
        Args:
            task_type: 任务类型，如"text_classification"
            model_name: 模型名称，用于加载相应的tokenizer或feature_extractor
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model_name = model_name
        
        # 根据任务类型加载不同的数据处理器
        if task_type in ["text_classification"]:
            self.processor = AutoTokenizer.from_pretrained(model_name)
        elif task_type in ["image_classification", "object_detection"]:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
    def load(self, data_path: str, **kwargs):
        """
        加载数据
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            加载的数据
        """
        # 检查路径是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
            
        # 根据文件类型加载数据
        if data_path.endswith(".csv"):
            # 加载CSV文件
            df = pd.read_csv(data_path)
            return Dataset.from_pandas(df)
        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
            # 加载JSON文件
            return load_dataset("json", data_files=data_path)
        elif os.path.isdir(data_path):
            # 尝试加载Hugging Face数据集目录
            try:
                return load_dataset(data_path)
            except:
                # 如果不是标准数据集格式，尝试其他加载方式
                files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
                if any(f.endswith(".csv") for f in files):
                    csv_files = [f for f in files if f.endswith(".csv")]
                    df = pd.concat([pd.read_csv(f) for f in csv_files])
                    return Dataset.from_pandas(df)
                elif any(f.endswith(".json") or f.endswith(".jsonl") for f in files):
                    json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
                    return load_dataset("json", data_files=json_files)
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到CSV或JSON文件: {data_path}")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
    def preprocess(self, data, **kwargs):
        """
        预处理数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        if self.task_type == "text_classification":
            # 文本分类预处理
            text_column = kwargs.get("text_column", "text")
            label_column = kwargs.get("label_column", "label")
            max_length = kwargs.get("max_length", 128)
            
            def tokenize_function(examples):
                return self.processor(
                    examples[text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
            # 应用tokenizer
            tokenized_data = data.map(tokenize_function, batched=True)
            
            # 重命名label列
            if label_column != "label" and label_column in data.column_names:
                tokenized_data = tokenized_data.rename_column(label_column, "label")
                
            return tokenized_data
            
        elif self.task_type == "image_classification":
            # 图像分类预处理
            image_column = kwargs.get("image_column", "image")
            label_column = kwargs.get("label_column", "label")
            
            def process_images(examples):
                images = examples[image_column]
                processed_images = self.processor(images=images, return_tensors="pt")
                return processed_images
                
            # 应用图像处理器
            processed_data = data.map(process_images, batched=True)
            
            # 重命名label列
            if label_column != "label" and label_column in data.column_names:
                processed_data = processed_data.rename_column(label_column, "label")
                
            return processed_data
            
        elif self.task_type == "object_detection":
            # 目标检测预处理
            image_column = kwargs.get("image_column", "image")
            bbox_column = kwargs.get("bbox_column", "bbox")
            label_column = kwargs.get("label_column", "label")
            
            def process_detection_data(examples):
                images = examples[image_column]
                bboxes = examples[bbox_column]
                labels = examples[label_column]
                
                processed_images = []
                for image, bbox, label in zip(images, bboxes, labels):
                    processed_image = self.processor(
                        images=image, 
                        annotations={
                            "boxes": bbox,
                            "labels": label
                        },
                        return_tensors="pt"
                    )
                    processed_images.append(processed_image)
                    
                return {"pixel_values": processed_images, "labels": labels}
                
            # 应用目标检测处理器
            processed_data = data.map(process_detection_data, batched=True)
            
            return processed_data
            
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
        
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理数据的便捷方法
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        # 加载数据
        data = self.load(data_path, **kwargs)
        
        # 预处理数据
        processed_data = self.preprocess(data, **kwargs)
        
        return processed_data
        
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
        # 确保数据是Dataset类型
        if not isinstance(data, Dataset):
            raise TypeError("data必须是datasets.Dataset类型")
            
        # 创建数据加载器
        dataloader = data.to_tf_dataset(
            columns=[c for c in data.column_names if c != "label"],
            label_cols=["label"] if "label" in data.column_names else None,
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        return dataloader 