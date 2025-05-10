from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator, Union


class BaseDataLoader(ABC):
    """数据加载器基类，定义了数据加载接口"""
    
    def __init__(self, **kwargs):
        """
        初始化数据加载器
        
        Args:
            **kwargs: 配置参数
        """
        self.config = kwargs
        
    @abstractmethod
    def load(self, data_path: str, **kwargs):
        """
        加载数据
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            加载的数据
        """
        pass
        
    @abstractmethod
    def preprocess(self, data, **kwargs):
        """
        预处理数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        pass
        
    @abstractmethod
    def process_data(self, data_path: str, **kwargs):
        """
        加载并处理数据的便捷方法
        
        Args:
            data_path: 数据路径
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        pass
        
    @abstractmethod
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
        pass


class TextClassificationDataLoader(BaseDataLoader):
    """文本分类数据加载器，处理text_cls_data.json格式的数据"""
    
    def __init__(self, **kwargs):
        """
        初始化文本分类数据加载器
        
        Args:
            **kwargs: 配置参数，可包含：
                - text_column: 文本列名，默认为"text"
                - label_column: 标签列名，默认为"label"
                - max_length: 最大文本长度，默认为128
        """
        super().__init__(**kwargs)
        self.text_column = kwargs.get("text_column", "text")
        self.label_column = kwargs.get("label_column", "label")
        self.max_length = kwargs.get("max_length", 128)
    
    @abstractmethod
    def tokenize(self, texts: List[str], **kwargs) -> Dict:
        """
        对文本进行分词
        
        Args:
            texts: 文本列表
            **kwargs: 分词参数
            
        Returns:
            Dict: 分词结果
        """
        pass


class ImageClassificationDataLoader(BaseDataLoader):
    """图像分类数据加载器，处理img_cls_data.json格式的数据"""
    
    def __init__(self, **kwargs):
        """
        初始化图像分类数据加载器
        
        Args:
            **kwargs: 配置参数，可包含：
                - image_column: 图像列名或路径，默认为"image"
                - label_column: 标签列名，默认为"label"
                - image_size: 图像大小，默认为(224, 224)
        """
        super().__init__(**kwargs)
        self.image_column = kwargs.get("image_column", "image")
        self.label_column = kwargs.get("label_column", "label")
        self.image_size = kwargs.get("image_size", (224, 224))
    
    @abstractmethod
    def process_images(self, images: List[Union[str, Any]], **kwargs) -> Dict:
        """
        处理图像
        
        Args:
            images: 图像路径列表或图像数据列表
            **kwargs: 图像处理参数
            
        Returns:
            Dict: 处理后的图像数据
        """
        pass


class ObjectDetectionDataLoader(BaseDataLoader):
    """目标检测数据加载器，处理img_det_data.json格式的数据"""
    
    def __init__(self, **kwargs):
        """
        初始化目标检测数据加载器
        
        Args:
            **kwargs: 配置参数，可包含：
                - image_column: 图像列名或路径，默认为"image"
                - boxes_column: 边界框列名，默认为"boxes"
                - labels_column: 标签列名，默认为"labels"
                - image_size: 图像大小，默认为(640, 640)
        """
        super().__init__(**kwargs)
        self.image_column = kwargs.get("image_column", "image")
        self.boxes_column = kwargs.get("boxes_column", "boxes")
        self.labels_column = kwargs.get("labels_column", "labels")
        self.image_size = kwargs.get("image_size", (640, 640))
    
    @abstractmethod
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
        pass 