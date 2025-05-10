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


class PaddleTextClassificationDataLoader(TextClassificationDataLoader):
    """PaddlePaddle的文本分类数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化PaddlePaddle文本分类数据加载器
        
        Args:
            model_name: 模型名称
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        
        # 尝试导入PaddleNLP
        try:
            import paddlenlp
            from paddlenlp.transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            print(f"警告: 无法导入PaddleNLP，请确保已安装相应的依赖")
            self.tokenizer = None
        
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
                
            # 转换为列表格式
            texts = [item.get(self.text_column, "") for item in data]
            labels = [item.get(self.label_column, 0) for item in data]
            
            return {
                "texts": texts,
                "labels": labels
            }
        elif data_path.endswith(".csv"):
            # 加载CSV文件
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # 提取文本和标签
            texts = df[self.text_column].tolist() if self.text_column in df.columns else []
            labels = df[self.label_column].tolist() if self.label_column in df.columns else []
            
            return {
                "texts": texts,
                "labels": labels
            }
        elif os.path.isdir(data_path):
            # 尝试加载目录中的文件
            json_files = [f for f in os.listdir(data_path) if f.endswith(".json") or f.endswith(".jsonl")]
            if json_files:
                all_data = []
                for file in json_files:
                    with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                        all_data.extend(json.load(f))
                
                texts = [item.get(self.text_column, "") for item in all_data]
                labels = [item.get(self.label_column, 0) for item in all_data]
                
                return {
                    "texts": texts,
                    "labels": labels
                }
            else:
                # 尝试加载CSV文件
                csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
                if csv_files:
                    import pandas as pd
                    all_data = []
                    for file in csv_files:
                        df = pd.read_csv(os.path.join(data_path, file))
                        all_data.append(df)
                    
                    combined_df = pd.concat(all_data, ignore_index=True)
                    texts = combined_df[self.text_column].tolist() if self.text_column in combined_df.columns else []
                    labels = combined_df[self.label_column].tolist() if self.label_column in combined_df.columns else []
                    
                    return {
                        "texts": texts,
                        "labels": labels
                    }
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON或CSV文件: {data_path}")
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
        if self.tokenizer is None:
            raise ValueError("Tokenizer未初始化，请确保PaddleNLP已正确安装")
            
        max_length = kwargs.get("max_length", self.max_length)
        
        return self.tokenizer(
            texts,
            max_seq_len=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
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
        if not isinstance(data, dict) or "texts" not in data or "labels" not in data:
            raise ValueError("数据格式不正确，应包含'texts'和'labels'字段")
            
        texts = data["texts"]
        labels = data["labels"]
        
        # 对文本进行分词
        tokenized = self.tokenize(texts, **kwargs)
        
        # 创建数据集
        import paddle
        
        # 转换为Paddle的Tensor
        input_ids = paddle.to_tensor(tokenized["input_ids"])
        token_type_ids = paddle.to_tensor(tokenized["token_type_ids"])
        attention_mask = paddle.to_tensor(tokenized["attention_mask"])
        labels = paddle.to_tensor(labels, dtype="int64")
        
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
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
        import paddle
        from paddle.io import Dataset, DataLoader
        
        class TextDataset(Dataset):
            def __init__(self, data):
                self.input_ids = data["input_ids"]
                self.token_type_ids = data["token_type_ids"]
                self.attention_mask = data["attention_mask"]
                self.labels = data["labels"]
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    "input_ids": self.input_ids[idx],
                    "token_type_ids": self.token_type_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx]
                }
                
        dataset = TextDataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )


class PaddleImageClassificationDataLoader(ImageClassificationDataLoader):
    """PaddlePaddle的图像分类数据加载器"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化PaddlePaddle图像分类数据加载器
        
        Args:
            model_name: 模型名称
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        
        # 尝试导入PaddlePaddle
        try:
            import paddle
            import paddle.vision.transforms as T
            
            # 定义图像变换
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            print(f"警告: 无法导入PaddlePaddle，请确保已安装相应的依赖")
            self.transform = None
        
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
                
            # 转换为列表格式
            image_paths = [item.get(self.image_column, "") for item in data]
            labels = [item.get(self.label_column, 0) for item in data]
            
            return {
                "images": image_paths,
                "labels": labels
            }
        elif os.path.isdir(data_path):
            # 尝试作为图像目录加载
            if os.path.exists(os.path.join(data_path, "images")) and os.path.exists(os.path.join(data_path, "labels.txt")):
                # PaddleClas格式的数据目录
                image_dir = os.path.join(data_path, "images")
                label_file = os.path.join(data_path, "labels.txt")
                
                # 加载标签映射
                label_map = {}
                with open(label_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        label_map[line.strip()] = i
                
                # 查找所有图像文件
                image_files = []
                for root, _, files in os.walk(image_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(os.path.join(root, file))
                
                # 提取标签
                labels = []
                for img_path in image_files:
                    # 从路径中提取类别名
                    relative_path = os.path.relpath(img_path, image_dir)
                    class_name = relative_path.split(os.sep)[0]
                    
                    # 查找对应的标签ID
                    label = label_map.get(class_name, 0)
                    labels.append(label)
                
                return {
                    "images": image_files,
                    "labels": labels
                }
            else:
                # 尝试加载JSON文件
                json_files = [f for f in os.listdir(data_path) if f.endswith(".json") or f.endswith(".jsonl")]
                if json_files:
                    all_data = []
                    for file in json_files:
                        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                            all_data.extend(json.load(f))
                    
                    image_paths = [item.get(self.image_column, "") for item in all_data]
                    labels = [item.get(self.label_column, 0) for item in all_data]
                    
                    return {
                        "images": image_paths,
                        "labels": labels
                    }
                else:
                    raise ValueError(f"不支持的数据格式，目录中未找到JSON文件或PaddleClas格式的数据: {data_path}")
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
        if self.transform is None:
            raise ValueError("图像变换未初始化，请确保PaddlePaddle已正确安装")
            
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
        processed_images = [self.transform(img) for img in loaded_images]
        
        import paddle
        return paddle.stack(processed_images)
        
    def preprocess(self, data, **kwargs):
        """
        预处理图像分类数据
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        if not isinstance(data, dict) or "images" not in data or "labels" not in data:
            raise ValueError("数据格式不正确，应包含'images'和'labels'字段")
            
        images = data["images"]
        labels = data["labels"]
        
        # 加载并处理图像
        loaded_images = []
        valid_labels = []
        
        for img_path, label in zip(images, labels):
            if isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    loaded_images.append(img)
                    valid_labels.append(label)
                except Exception as e:
                    print(f"警告: 无法加载图像 {img_path}: {e}")
            elif hasattr(img_path, "mode"):
                # 如果已经是PIL.Image对象
                loaded_images.append(img_path)
                valid_labels.append(label)
        
        # 处理图像
        import paddle
        processed_images = [self.transform(img) for img in loaded_images]
        processed_images = paddle.stack(processed_images) if processed_images else paddle.zeros((0, 3, *self.image_size))
        
        # 转换标签
        labels = paddle.to_tensor(valid_labels, dtype="int64")
        
        return {
            "images": processed_images,
            "labels": labels
        }
        
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
        import paddle
        from paddle.io import Dataset, DataLoader
        
        class ImageDataset(Dataset):
            def __init__(self, data):
                self.images = data["images"]
                self.labels = data["labels"]
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return {
                    "images": self.images[idx],
                    "labels": self.labels[idx]
                }
                
        dataset = ImageDataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        ) 