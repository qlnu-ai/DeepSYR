import os
import json
from typing import Dict, Any, Optional, List, Iterator, Union
import pandas as pd
import numpy as np
from PIL import Image

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

from core.dataloader import (
    BaseDataLoader, 
    TextClassificationDataLoader,
    ImageClassificationDataLoader, 
    ObjectDetectionDataLoader
)


class TransformersTextClassificationDataLoader(TextClassificationDataLoader):
    """Transformers库的文本分类数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化Transformers文本分类数据加载器
        
        Args:
            model_name: 模型名称，用于加载相应的tokenizer
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load(self, data_path: str, **kwargs):
        """
        加载文本分类数据
        
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
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            # 直接加载JSON文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 转换为Dataset格式
            texts = [item.get(self.text_column, "") for item in data]
            labels = [item.get(self.label_column, 0) for item in data]
            
            return Dataset.from_dict({
                self.text_column: texts,
                self.label_column: labels
            })
        elif data_path.endswith(".csv"):
            # 加载CSV文件
            df = pd.read_csv(data_path)
            return Dataset.from_pandas(df)
        elif os.path.isdir(data_path):
            # 尝试加载目录中的文件
            try:
                return load_dataset(data_path)
            except:
                # 如果不是标准数据集格式，尝试其他加载方式
                files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
                json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
                if json_files:
                    all_data = []
                    for file in json_files:
                        with open(file, 'r', encoding='utf-8') as f:
                            all_data.extend(json.load(f))
                    
                    texts = [item.get(self.text_column, "") for item in all_data]
                    labels = [item.get(self.label_column, 0) for item in all_data]
                    
                    return Dataset.from_dict({
                        self.text_column: texts,
                        self.label_column: labels
                    })
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON文件: {data_path}")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
    def tokenize(self, texts: List[str], **kwargs) -> Dict:
        """
        对文本进行分词
        
        Args:
            texts: 文本列表
            **kwargs: 分词参数
            
        Returns:
            Dict: 分词结果
        """
        max_length = kwargs.get("max_length", self.max_length)
        
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
    def preprocess(self, data, **kwargs):
        """
        预处理文本分类数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        # 确保数据是Dataset类型
        if not isinstance(data, Dataset):
            raise TypeError("data必须是datasets.Dataset类型")
            
        # 定义分词函数
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
        # 应用tokenizer
        tokenized_data = data.map(tokenize_function, batched=True)
        
        # 重命名label列
        if self.label_column != "label" and self.label_column in data.column_names:
            tokenized_data = tokenized_data.rename_column(self.label_column, "label")
            
        return tokenized_data
        
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理文本分类数据
        
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


class TransformersImageClassificationDataLoader(ImageClassificationDataLoader):
    """Transformers库的图像分类数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化Transformers图像分类数据加载器
        
        Args:
            model_name: 模型名称，用于加载相应的图像处理器
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
    def load(self, data_path: str, **kwargs):
        """
        加载图像分类数据
        
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
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            # 直接加载JSON文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 转换为Dataset格式
            image_paths = [item.get(self.image_column, "") for item in data]
            labels = [item.get(self.label_column, 0) for item in data]
            
            return Dataset.from_dict({
                self.image_column: image_paths,
                self.label_column: labels
            })
        elif os.path.isdir(data_path):
            # 尝试加载目录中的文件
            try:
                return load_dataset(data_path)
            except:
                # 如果不是标准数据集格式，尝试其他加载方式
                files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
                json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
                if json_files:
                    all_data = []
                    for file in json_files:
                        with open(file, 'r', encoding='utf-8') as f:
                            all_data.extend(json.load(f))
                    
                    image_paths = [item.get(self.image_column, "") for item in all_data]
                    labels = [item.get(self.label_column, 0) for item in all_data]
                    
                    return Dataset.from_dict({
                        self.image_column: image_paths,
                        self.label_column: labels
                    })
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON文件: {data_path}")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
    def process_images(self, images: List[Union[str, Any]], **kwargs) -> Dict:
        """
        处理图像
        
        Args:
            images: 图像路径列表或图像数据列表
            **kwargs: 图像处理参数
            
        Returns:
            Dict: 处理后的图像数据
        """
        # 加载图像
        loaded_images = []
        for img in images:
            if isinstance(img, str) and os.path.exists(img):
                # 如果是路径，加载图像
                loaded_images.append(Image.open(img).convert("RGB"))
            elif hasattr(img, "mode"):
                # 如果已经是PIL.Image对象
                loaded_images.append(img)
            else:
                raise ValueError(f"不支持的图像格式: {type(img)}")
                
        # 处理图像
        return self.processor(images=loaded_images, return_tensors="pt")
        
    def preprocess(self, data, **kwargs):
        """
        预处理图像分类数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        # 确保数据是Dataset类型
        if not isinstance(data, Dataset):
            raise TypeError("data必须是datasets.Dataset类型")
            
        # 定义图像处理函数
        def process_image_batch(examples):
            images = []
            for img_path in examples[self.image_column]:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                else:
                    # 如果图像不存在，使用空白图像
                    images.append(Image.new("RGB", self.image_size, (255, 255, 255)))
                    
            # 处理图像
            processed = self.processor(images=images, return_tensors="pt")
            
            # 将处理结果添加到examples中
            for key, value in processed.items():
                examples[key] = value
                
            return examples
            
        # 应用图像处理器
        processed_data = data.map(process_image_batch, batched=True)
        
        # 重命名label列
        if self.label_column != "label" and self.label_column in data.column_names:
            processed_data = processed_data.rename_column(self.label_column, "label")
            
        return processed_data
        
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理图像分类数据
        
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
        columns_to_remove = [self.image_column]  # 移除原始图像路径列
        if self.label_column != "label":
            columns_to_remove.append(self.label_column)
            
        dataloader = data.to_tf_dataset(
            columns=[c for c in data.column_names if c not in columns_to_remove],
            label_cols=["label"] if "label" in data.column_names else None,
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        return dataloader


class TransformersObjectDetectionDataLoader(ObjectDetectionDataLoader):
    """Transformers库的目标检测数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化Transformers目标检测数据加载器
        
        Args:
            model_name: 模型名称，用于加载相应的图像处理器
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        
    def load(self, data_path: str, **kwargs):
        """
        加载目标检测数据
        
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
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            # 直接加载JSON文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 转换为Dataset格式
            image_paths = [item.get(self.image_column, "") for item in data]
            boxes = [item.get(self.boxes_column, []) for item in data]
            labels = [item.get(self.labels_column, []) for item in data]
            
            return Dataset.from_dict({
                self.image_column: image_paths,
                self.boxes_column: boxes,
                self.labels_column: labels
            })
        elif os.path.isdir(data_path):
            # 尝试加载目录中的文件
            try:
                return load_dataset(data_path)
            except:
                # 如果不是标准数据集格式，尝试其他加载方式
                files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
                json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
                if json_files:
                    all_data = []
                    for file in json_files:
                        with open(file, 'r', encoding='utf-8') as f:
                            all_data.extend(json.load(f))
                    
                    image_paths = [item.get(self.image_column, "") for item in all_data]
                    boxes = [item.get(self.boxes_column, []) for item in all_data]
                    labels = [item.get(self.labels_column, []) for item in all_data]
                    
                    return Dataset.from_dict({
                        self.image_column: image_paths,
                        self.boxes_column: boxes,
                        self.labels_column: labels
                    })
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON文件: {data_path}")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
    def process_detection_data(self, images: List[Union[str, Any]], boxes: List[List], labels: List[List], **kwargs) -> Dict:
        """
        处理目标检测数据
        
        Args:
            images: 图像路径列表或图像数据列表
            boxes: 边界框列表，每个元素是一个图像中的所有边界框
            labels: 标签列表，每个元素是一个图像中的所有标签
            **kwargs: 处理参数
            
        Returns:
            Dict: 处理后的目标检测数据
        """
        # 加载图像
        loaded_images = []
        for img in images:
            if isinstance(img, str) and os.path.exists(img):
                # 如果是路径，加载图像
                loaded_images.append(Image.open(img).convert("RGB"))
            elif hasattr(img, "mode"):
                # 如果已经是PIL.Image对象
                loaded_images.append(img)
            else:
                raise ValueError(f"不支持的图像格式: {type(img)}")
                
        # 处理图像和标注
        processed_data = []
        for img, bbox, label in zip(loaded_images, boxes, labels):
            # 将中心点坐标转换为左上角和右下角坐标
            transformed_boxes = []
            for box in bbox:
                x_center, y_center, width, height = box
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                transformed_boxes.append([x1, y1, x2, y2])
                
            # 处理图像和标注
            processed = self.processor(
                images=img,
                annotations={
                    "boxes": transformed_boxes,
                    "labels": label
                },
                return_tensors="pt"
            )
            processed_data.append(processed)
            
        return processed_data
        
    def preprocess(self, data, **kwargs):
        """
        预处理目标检测数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        # 确保数据是Dataset类型
        if not isinstance(data, Dataset):
            raise TypeError("data必须是datasets.Dataset类型")
            
        # 定义目标检测数据处理函数
        def process_detection_batch(examples):
            images = []
            all_boxes = []
            all_labels = []
            
            # 加载图像和标注
            for img_path, boxes, labels in zip(
                examples[self.image_column], 
                examples[self.boxes_column], 
                examples[self.labels_column]
            ):
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    all_boxes.append(boxes)
                    all_labels.append(labels)
                else:
                    # 如果图像不存在，使用空白图像和空标注
                    images.append(Image.new("RGB", self.image_size, (255, 255, 255)))
                    all_boxes.append([])
                    all_labels.append([])
                    
            # 处理图像和标注
            processed_batch = self.process_detection_data(images, all_boxes, all_labels)
            
            # 将处理结果添加到examples中
            examples["processed_data"] = processed_batch
            
            return examples
            
        # 应用目标检测处理器
        processed_data = data.map(process_detection_batch, batched=True)
        
        return processed_data
        
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理目标检测数据
        
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
        columns_to_keep = ["processed_data"]
        if self.labels_column in data.column_names:
            columns_to_keep.append(self.labels_column)
            
        dataloader = data.to_tf_dataset(
            columns=columns_to_keep,
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        return dataloader 