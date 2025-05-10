import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable, Dict, Any, Optional, List, Tuple

from utils.logging import logger


def setup_distributed(rank: int, world_size: int) -> None:
    """
    设置分布式环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
    """
    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    logger.info(f"分布式环境已设置，rank: {rank}, world_size: {world_size}")


def cleanup_distributed() -> None:
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed(
    fn: Callable,
    world_size: int,
    fn_args: Tuple = (),
    fn_kwargs: Dict[str, Any] = {}
) -> None:
    """
    运行分布式训练
    
    Args:
        fn: 训练函数
        world_size: 总进程数
        fn_args: 训练函数的位置参数
        fn_kwargs: 训练函数的关键字参数
    """
    mp.spawn(
        _distributed_worker,
        args=(fn, world_size, fn_args, fn_kwargs),
        nprocs=world_size,
        join=True
    )


def _distributed_worker(
    rank: int,
    fn: Callable,
    world_size: int,
    fn_args: Tuple,
    fn_kwargs: Dict[str, Any]
) -> None:
    """
    分布式工作进程
    
    Args:
        rank: 当前进程的rank
        fn: 训练函数
        world_size: 总进程数
        fn_args: 训练函数的位置参数
        fn_kwargs: 训练函数的关键字参数
    """
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置随机种子
    torch.manual_seed(42 + rank)
    
    # 调用训练函数
    try:
        fn(rank, world_size, *fn_args, **fn_kwargs)
    except Exception as e:
        logger.error(f"进程 {rank} 发生错误: {e}")
        raise
    finally:
        # 清理分布式环境
        cleanup_distributed()


class DistributedTrainer:
    """分布式训练器，封装分布式训练的常用操作"""
    
    def __init__(self, model, optimizer, device, rank: int = 0, world_size: int = 1):
        """
        初始化分布式训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            device: 设备
            rank: 当前进程的rank
            world_size: 总进程数
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # 如果是分布式训练，包装模型
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank
            )
            
    def train_step(self, batch, loss_fn):
        """
        训练一个批次
        
        Args:
            batch: 批次数据
            loss_fn: 损失函数
            
        Returns:
            Dict[str, float]: 训练结果
        """
        # 将数据移动到设备上
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 前向传播
        outputs = self.model(**batch)
        
        # 计算损失
        loss = loss_fn(outputs, batch)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def validate_step(self, batch, loss_fn):
        """
        验证一个批次
        
        Args:
            batch: 批次数据
            loss_fn: 损失函数
            
        Returns:
            Dict[str, float]: 验证结果
        """
        # 将数据移动到设备上
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**batch)
            
            # 计算损失
            loss = loss_fn(outputs, batch)
            
        return {"loss": loss.item()}
    
    def save_checkpoint(self, save_path: str, epoch: int, step: int, metrics: Optional[Dict[str, float]] = None):
        """
        保存检查点
        
        Args:
            save_path: 保存路径
            epoch: 当前轮次
            step: 当前步数
            metrics: 评估指标
            
        Returns:
            str: 检查点路径
        """
        # 只在主进程保存检查点
        if self.rank != 0:
            return None
            
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备检查点数据
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
            
        # 保存检查点
        torch.save(checkpoint, save_path)
        logger.info(f"保存检查点到 {save_path}")
        
        return save_path
    
    def load_checkpoint(self, load_path: str, strict: bool = True):
        """
        加载检查点
        
        Args:
            load_path: 加载路径
            strict: 是否严格加载模型参数
            
        Returns:
            Dict[str, Any]: 检查点数据
        """
        # 检查文件是否存在
        if not os.path.exists(load_path):
            logger.warning(f"检查点文件不存在: {load_path}")
            return {}
            
        # 加载检查点
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载模型参数
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            
        # 加载优化器参数
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        logger.info(f"加载检查点从 {load_path}")
        
        return checkpoint
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        在所有进程间归约张量
        
        Args:
            tensor: 张量
            op: 操作，可选"sum"、"mean"、"max"、"min"
            
        Returns:
            torch.Tensor: 归约后的张量
        """
        if self.world_size <= 1:
            return tensor
            
        # 选择操作
        if op == "sum":
            reduce_op = dist.ReduceOp.SUM
        elif op == "mean":
            reduce_op = dist.ReduceOp.SUM
        elif op == "max":
            reduce_op = dist.ReduceOp.MAX
        elif op == "min":
            reduce_op = dist.ReduceOp.MIN
        else:
            raise ValueError(f"不支持的操作: {op}")
            
        # 归约
        dist.all_reduce(tensor, reduce_op)
        
        # 如果是平均操作，除以进程数
        if op == "mean":
            tensor /= self.world_size
            
        return tensor
    
    def is_main_process(self) -> bool:
        """
        是否是主进程
        
        Returns:
            bool: 是否是主进程
        """
        return self.rank == 0
    
    def barrier(self) -> None:
        """同步所有进程"""
        if self.world_size > 1:
            dist.barrier() 