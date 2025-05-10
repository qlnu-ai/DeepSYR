#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch

from core.factory import Factory
from utils import logger, setup_logger, CheckpointManager, MetricsCalculator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文本分类任务运行脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--train", action="store_true", help="是否训练模型")
    parser.add_argument("--eval", action="store_true", help="是否评估模型")
    parser.add_argument("--predict", help="预测输入文本或包含文本的文件路径")
    parser.add_argument("--model_path", help="模型路径，用于评估或预测")
    parser.add_argument("--output_dir", help="输出目录，覆盖配置文件中的设置")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置日志
    log_dir = config.get("training", {}).get("logging_dir")
    setup_logger("DeepSYR", log_dir=log_dir)
    
    # 打印配置信息
    logger.info(f"任务类型: {config['task_type']}")
    logger.info(f"后端框架: {config['model_type']}")
    logger.info(f"模型名称: {config['model_name']}")
    logger.info(f"设备: {args.device}")
    
    # 覆盖输出目录
    if args.output_dir:
        config["training"] = config.get("training", {})
        config["training"]["output_dir"] = args.output_dir
    
    # 创建任务
    task = Factory.create_task(
        task_type=config["task_type"],
        backend=config["model_type"],
        model_name=config["model_name"],
        num_labels=config.get("num_labels", 2),
        **config
    )
    
    # 根据命令行参数执行相应操作
    if args.train:
        # 训练模型
        logger.info("开始训练模型...")
        
        # 准备数据
        train_data = task.prepare_data(
            config["data"]["train_path"],
            **config["data"]
        )
        
        # 如果有评估数据，也准备评估数据
        eval_data = None
        if "eval_path" in config["data"]:
            eval_data = task.prepare_data(
                config["data"]["eval_path"],
                **config["data"]
            )
            
        # 准备模型
        task.prepare_model()
        
        # 训练模型
        training_config = config.get("training", {})
        task.train(
            train_data,
            eval_data,
            **training_config
        )
        
        # 保存模型
        output_dir = training_config.get("output_dir", "./outputs")
        task.save(output_dir)
        logger.info(f"模型已保存到 {output_dir}")
        
    if args.eval:
        # 评估模型
        logger.info("开始评估模型...")
        
        # 如果指定了模型路径，加载模型
        if args.model_path:
            task.load(args.model_path)
            
        # 准备评估数据
        eval_data = task.prepare_data(
            config["data"]["eval_path"],
            **config["data"]
        )
        
        # 评估模型
        metrics = task.evaluate(eval_data)
        
        # 打印评估结果
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value}")
            
    if args.predict:
        # 预测
        logger.info("开始预测...")
        
        # 如果指定了模型路径，加载模型
        if args.model_path:
            task.load(args.model_path)
            
        # 准备输入数据
        if os.path.exists(args.predict):
            # 如果是文件路径，读取文件
            with open(args.predict, "r", encoding="utf-8") as f:
                texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
        else:
            # 否则认为是直接输入的文本
            texts = [args.predict]
            
        # 进行预测
        results = task.predict(texts)
        
        # 打印预测结果
        for text, result in zip(texts, results):
            if "label" in result:
                logger.info(f"文本: {text}")
                logger.info(f"预测类别: {result['label']}")
                logger.info(f"置信度: {max(result['probabilities']):.4f}")
                logger.info("-" * 50)
            else:
                logger.info(f"文本: {text}")
                logger.info(f"预测类别ID: {result['class']}")
                logger.info(f"置信度: {max(result['probabilities']):.4f}")
                logger.info("-" * 50)
    
    logger.info("完成!")


if __name__ == "__main__":
    main() 