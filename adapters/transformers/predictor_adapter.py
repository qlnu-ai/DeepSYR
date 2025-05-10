import os
from typing import Dict, Any, Optional, List, Union
import numpy as np

from core.predictor import BasePredictor


class TransformersPredictorAdapter(BasePredictor):
    """Transformers库的预测适配器"""
    
    def __init__(self, model, task_type: str, **kwargs):
        """
        初始化Transformers预测适配器
        
        Args:
            model: transformers模型实例
            task_type: 任务类型
            **kwargs: 其他参数
        """
        super().__init__(model, **kwargs)
        self.task_type = task_type
        
        # 根据任务类型加载不同的处理器
        if task_type == "text_classification":
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(kwargs.get("model_name", "bert-base-uncased"))
        elif task_type in ["image_classification", "object_detection"]:
            from transformers import AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(kwargs.get("model_name", "google/vit-base-patch16-224"))
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
            
    def preprocess(self, input_data, **kwargs):
        """
        预处理输入数据
        
        Args:
            input_data: 输入数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        if self.task_type == "text_classification":
            # 文本分类预处理
            max_length = kwargs.get("max_length", 128)
            
            if isinstance(input_data, str):
                # 单个文本
                return self.processor(
                    input_data,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                # 多个文本
                return self.processor(
                    input_data,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
            else:
                raise TypeError("text_classification的input_data必须是字符串或字符串列表")
                
        elif self.task_type == "image_classification":
            # 图像分类预处理
            from PIL import Image
            
            if isinstance(input_data, str) and os.path.exists(input_data):
                # 图像路径
                image = Image.open(input_data).convert("RGB")
                return self.processor(images=image, return_tensors="pt")
            elif hasattr(input_data, "mode"):  # 检查是否是PIL.Image
                # PIL图像
                return self.processor(images=input_data, return_tensors="pt")
            else:
                raise TypeError("image_classification的input_data必须是图像路径或PIL.Image对象")
                
        elif self.task_type == "object_detection":
            # 目标检测预处理
            from PIL import Image
            
            if isinstance(input_data, str) and os.path.exists(input_data):
                # 图像路径
                image = Image.open(input_data).convert("RGB")
                return self.processor(images=image, return_tensors="pt")
            elif hasattr(input_data, "mode"):  # 检查是否是PIL.Image
                # PIL图像
                return self.processor(images=input_data, return_tensors="pt")
            else:
                raise TypeError("object_detection的input_data必须是图像路径或PIL.Image对象")
                
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
            
    def predict(self, input_data, **kwargs):
        """
        使用模型进行预测
        
        Args:
            input_data: 输入数据
            **kwargs: 预测参数
            
        Returns:
            预测结果
        """
        # 预处理数据
        processed_data = self.preprocess(input_data, **kwargs)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 进行预测
        with torch_no_grad():
            outputs = self.model(**processed_data)
            
        # 后处理输出
        results = self.postprocess(outputs, **kwargs)
        
        return results
        
    def postprocess(self, model_output, **kwargs):
        """
        后处理模型输出
        
        Args:
            model_output: 模型输出
            **kwargs: 后处理参数
            
        Returns:
            后处理后的结果
        """
        import torch
        
        if self.task_type == "text_classification":
            # 文本分类后处理
            logits = model_output.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # 获取预测类别和概率
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # 转换为Python对象
            result = {
                "class": predicted_class.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist()
            }
            
            # 如果提供了id2label映射，转换类别ID为标签
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                if isinstance(result["class"], list):
                    result["label"] = [id2label[cls_id] for cls_id in result["class"]]
                else:
                    result["label"] = id2label[result["class"]]
                    
            return result
            
        elif self.task_type == "image_classification":
            # 图像分类后处理
            logits = model_output.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # 获取预测类别和概率
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # 转换为Python对象
            result = {
                "class": predicted_class.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist()
            }
            
            # 如果提供了id2label映射，转换类别ID为标签
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                if isinstance(result["class"], list):
                    result["label"] = [id2label[cls_id] for cls_id in result["class"]]
                else:
                    result["label"] = id2label[result["class"]]
                    
            return result
            
        elif self.task_type == "object_detection":
            # 目标检测后处理
            scores = model_output.scores
            boxes = model_output.boxes
            labels = model_output.labels
            
            # 应用阈值过滤
            threshold = kwargs.get("threshold", 0.5)
            mask = scores > threshold
            
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            
            # 转换为Python对象
            result = {
                "boxes": filtered_boxes.cpu().numpy().tolist(),
                "scores": filtered_scores.cpu().numpy().tolist(),
                "labels": filtered_labels.cpu().numpy().tolist()
            }
            
            # 如果提供了id2label映射，转换类别ID为标签
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                result["label_names"] = [id2label[label_id] for label_id in result["labels"]]
                
            return result
            
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
            
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
        results = []
        
        # 按批次处理
        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i:i+batch_size]
            
            # 处理不同类型的输入
            if self.task_type == "text_classification" and all(isinstance(item, str) for item in batch):
                # 文本批量处理
                processed_batch = self.preprocess(batch, **kwargs)
                self.model.eval()
                
                with torch_no_grad():
                    outputs = self.model(**processed_batch)
                    
                batch_results = self.postprocess(outputs, **kwargs)
                results.extend(batch_results)
                
            else:
                # 单独处理每个样本
                for item in batch:
                    result = self.predict(item, **kwargs)
                    results.append(result)
                    
        return results
        
        
def torch_no_grad():
    """获取torch.no_grad上下文管理器"""
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        # 如果torch不可用，返回一个dummy上下文管理器
        class DummyContextManager:
            def __enter__(self): return None
            def __exit__(self, *args): return None
        return DummyContextManager() 