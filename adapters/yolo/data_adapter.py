import os
import json
from typing import Dict, Any, Optional, List, Iterator, Union
import numpy as np
from PIL import Image

from core.dataloader import (
    BaseDataLoader, 
    TextClassificationDataLoader,
    ImageClassificationDataLoader, 
    ObjectDetectionDataLoader
)


class YoloObjectDetectionDataLoader(ObjectDetectionDataLoader):
    """YOLO的目标检测数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化YOLO目标检测数据加载器
        
        Args:
            model_name: 模型名称，例如"yolov5s"、"yolov8n"等
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        
        # 根据模型名称确定YOLO版本
        if model_name.startswith("yolov5"):
            self.yolo_version = 5
        elif model_name.startswith("yolov8"):
            self.yolo_version = 8
        else:
            self.yolo_version = 5  # 默认使用YOLOv5
            
        # 尝试导入YOLO库
        try:
            if self.yolo_version == 8:
                from ultralytics import YOLO
                self.processor = YOLO(model_name)
            else:
                import torch
                self.processor = None  # YOLOv5不需要特定处理器
        except ImportError:
            self.processor = None
            print(f"警告: 无法导入YOLO库，请确保已安装相应的依赖")
        
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
                
            # 转换为字典格式
            image_paths = [item.get(self.image_column, "") for item in data]
            boxes = [item.get(self.boxes_column, []) for item in data]
            labels = [item.get(self.labels_column, []) for item in data]
            
            return {
                "images": image_paths,
                "boxes": boxes,
                "labels": labels
            }
        elif data_path.endswith(".yaml") or data_path.endswith(".yml"):
            # YOLO格式的数据集配置
            import yaml
            with open(data_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 返回YOLO配置
            return {
                "config": config,
                "yaml_path": data_path
            }
        elif os.path.isdir(data_path):
            # 尝试作为YOLO格式的数据目录加载
            if os.path.exists(os.path.join(data_path, "images")) and os.path.exists(os.path.join(data_path, "labels")):
                image_dir = os.path.join(data_path, "images")
                label_dir = os.path.join(data_path, "labels")
                
                # 找到所有图像文件
                image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
                
                image_paths = []
                boxes_list = []
                labels_list = []
                
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    image_paths.append(img_path)
                    
                    # 尝试加载对应的标签文件
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    label_path = os.path.join(label_dir, label_file)
                    
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            
                        # 解析YOLO格式的标签
                        boxes = []
                        labels = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                label = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                boxes.append([x_center, y_center, width, height])
                                labels.append(label)
                                
                        boxes_list.append(boxes)
                        labels_list.append(labels)
                    else:
                        # 如果没有找到标签文件，添加空标签
                        boxes_list.append([])
                        labels_list.append([])
                        
                return {
                    "images": image_paths,
                    "boxes": boxes_list,
                    "labels": labels_list
                }
            else:
                # 尝试作为普通图像目录加载
                json_files = [f for f in os.listdir(data_path) if f.endswith(".json") or f.endswith(".jsonl")]
                if json_files:
                    all_data = []
                    for file in json_files:
                        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                            all_data.extend(json.load(f))
                    
                    image_paths = [item.get(self.image_column, "") for item in all_data]
                    boxes = [item.get(self.boxes_column, []) for item in all_data]
                    labels = [item.get(self.labels_column, []) for item in all_data]
                    
                    return {
                        "images": image_paths,
                        "boxes": boxes,
                        "labels": labels
                    }
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON文件或YOLO格式的数据: {data_path}")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
    def process_detection_data(self, images: List[Union[str, Any]], boxes: List[List], labels: List[List], **kwargs) -> Dict:
        """
        处理目标检测数据
        
        Args:
            images: 图像路径列表或图像数据列表
            boxes: 边界框列表，格式为[[x_center, y_center, width, height], ...]
            labels: 标签列表
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
        for i, (img, bbox, label) in enumerate(zip(loaded_images, boxes, labels)):
            # 获取图像尺寸
            width, height = img.size
            
            # 将标注比例转换为像素坐标
            pixel_boxes = []
            for box in bbox:
                x_center, y_center, box_width, box_height = box
                
                # 比例转像素
                x_center_px = x_center * width
                y_center_px = y_center * height
                width_px = box_width * width
                height_px = box_height * height
                
                # 计算左上角和右下角坐标
                x1 = x_center_px - width_px / 2
                y1 = y_center_px - height_px / 2
                x2 = x_center_px + width_px / 2
                y2 = y_center_px + height_px / 2
                
                pixel_boxes.append([x1, y1, x2, y2])
                
            # 创建处理后的数据
            processed = {
                "image": img,
                "boxes": np.array(pixel_boxes) if pixel_boxes else np.zeros((0, 4)),
                "labels": np.array(label) if label else np.zeros(0),
                "image_id": i
            }
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
        # 如果是YOLO配置文件，直接返回
        if isinstance(data, dict) and "config" in data:
            return data
            
        # 否则处理加载的JSON数据
        images = data["images"]
        boxes = data["boxes"]
        labels = data["labels"]
        
        return self.process_detection_data(images, boxes, labels, **kwargs)
        
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
        # 如果是YOLO配置文件，使用YOLO的DataLoader
        if isinstance(data, dict) and "config" in data:
            if self.yolo_version == 8:
                from ultralytics.data.dataset import YOLODataset
                from torch.utils.data import DataLoader
                
                dataset = YOLODataset(
                    img_path=data.get("config", {}).get("path", ""),
                    data=data.get("yaml_path", "")
                )
                
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=kwargs.get("num_workers", 4),
                    pin_memory=True
                )
            else:
                # YOLOv5的DataLoader
                import torch
                from torch.utils.data import DataLoader
                
                # 使用YOLOv5的create_dataloader函数
                try:
                    from yolov5.utils.datasets import create_dataloader
                    return create_dataloader(
                        data.get("yaml_path", ""),
                        img_size=self.image_size[0],  # 图像大小
                        batch_size=batch_size,
                        augment=False,  # 不使用数据增强
                        rect=False,  # 不使用矩形训练
                        rank=-1
                    )[0]
                except ImportError:
                    print("警告：无法导入YOLOv5的create_dataloader函数，使用自定义DataLoader")
                    # 自定义简单DataLoader
                    class SimpleDataset(torch.utils.data.Dataset):
                        def __init__(self, images, boxes, labels):
                            self.images = images
                            self.boxes = boxes
                            self.labels = labels
                            
                        def __len__(self):
                            return len(self.images)
                            
                        def __getitem__(self, idx):
                            return {
                                "img": self.images[idx],
                                "boxes": self.boxes[idx],
                                "labels": self.labels[idx]
                            }
                            
                    return DataLoader(
                        SimpleDataset(
                            data.get("images", []),
                            data.get("boxes", []),
                            data.get("labels", [])
                        ),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=lambda x: x  # 保持批次作为列表
                    )
        
        # 对于处理过的数据，创建自定义DataLoader
        import torch
        from torch.utils.data import DataLoader
        
        class DetectionDataset(torch.utils.data.Dataset):
            def __init__(self, processed_data):
                self.data = processed_data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
                
        return DataLoader(
            DetectionDataset(data),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x  # 保持批次作为列表
        ) 