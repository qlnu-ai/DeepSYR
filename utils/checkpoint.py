import os
import json
import time
from typing import Dict, Any, Optional, Union
import torch

from utils.logging import logger


class CheckpointManager:
    """检查点管理器，提供模型检查点的保存和加载功能"""
    
    def __init__(self, save_dir: str, max_to_keep: int = 5):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            max_to_keep: 最多保留的检查点数量
        """
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载检查点信息
        self._load_checkpoint_info()
        
    def _load_checkpoint_info(self):
        """加载检查点信息"""
        info_path = os.path.join(self.save_dir, "checkpoint_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                logger.warning(f"加载检查点信息失败: {e}")
                self.checkpoints = []
        else:
            self.checkpoints = []
            
    def _save_checkpoint_info(self):
        """保存检查点信息"""
        info_path = os.path.join(self.save_dir, "checkpoint_info.json")
        try:
            with open(info_path, "w") as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.warning(f"保存检查点信息失败: {e}")
            
    def save(self, model, optimizer=None, epoch: int = 0, step: int = 0, 
             metrics: Optional[Dict[str, float]] = None, **kwargs) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前轮次
            step: 当前步数
            metrics: 评估指标
            **kwargs: 其他参数
            
        Returns:
            str: 检查点路径
        """
        # 生成检查点名称
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        
        # 准备检查点数据
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            
        if metrics is not None:
            checkpoint["metrics"] = metrics
            
        # 添加其他参数
        checkpoint.update(kwargs)
        
        # 保存检查点
        try:
            torch.save(checkpoint, ckpt_path)
            logger.info(f"保存检查点到 {ckpt_path}")
            
            # 更新检查点信息
            ckpt_info = {
                "name": ckpt_name,
                "path": ckpt_path,
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                "timestamp": timestamp
            }
            self.checkpoints.append(ckpt_info)
            
            # 如果检查点数量超过上限，删除最旧的检查点
            if len(self.checkpoints) > self.max_to_keep:
                oldest_ckpt = self.checkpoints.pop(0)
                oldest_path = oldest_ckpt["path"]
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                    logger.info(f"删除旧检查点: {oldest_path}")
                    
            # 保存检查点信息
            self._save_checkpoint_info()
            
            return ckpt_path
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return ""
            
    def load(self, model, ckpt_path: Optional[str] = None, optimizer=None, 
             load_optimizer: bool = True, strict: bool = True) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            model: 模型
            ckpt_path: 检查点路径，如果为None则加载最新的检查点
            optimizer: 优化器
            load_optimizer: 是否加载优化器状态
            strict: 是否严格加载模型参数
            
        Returns:
            Dict[str, Any]: 检查点数据
        """
        # 如果未指定检查点路径，加载最新的检查点
        if ckpt_path is None:
            if not self.checkpoints:
                logger.warning("没有可用的检查点")
                return {}
            ckpt_path = self.checkpoints[-1]["path"]
            
        # 检查文件是否存在
        if not os.path.exists(ckpt_path):
            logger.warning(f"检查点文件不存在: {ckpt_path}")
            return {}
            
        try:
            # 加载检查点
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
            # 加载模型参数
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info(f"加载模型参数从 {ckpt_path}")
            
            # 加载优化器参数
            if optimizer is not None and load_optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("加载优化器参数")
                
            return checkpoint
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return {}
            
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的检查点信息
        
        Returns:
            Optional[Dict[str, Any]]: 检查点信息
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]
        
    def get_best_checkpoint(self, metric_name: str, higher_better: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取最佳的检查点信息
        
        Args:
            metric_name: 指标名称
            higher_better: 是否越高越好
            
        Returns:
            Optional[Dict[str, Any]]: 检查点信息
        """
        if not self.checkpoints:
            return None
            
        best_ckpt = None
        best_value = float("-inf") if higher_better else float("inf")
        
        for ckpt in self.checkpoints:
            metrics = ckpt.get("metrics", {})
            if metric_name in metrics:
                value = metrics[metric_name]
                if (higher_better and value > best_value) or (not higher_better and value < best_value):
                    best_value = value
                    best_ckpt = ckpt
                    
        return best_ckpt 