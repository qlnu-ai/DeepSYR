import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)


class MetricsCalculator:
    """评估指标计算器，提供常用评估指标的计算功能"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_score: Optional[Union[List, np.ndarray]] = None,
        average: str = "macro"
    ) -> Dict[str, float]:
        """
        计算分类任务的评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_score: 预测概率或分数，用于计算AUC和AP
            average: 多分类指标的平均方式，可选"micro", "macro", "weighted"
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        metrics = {}
        
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 计算准确率
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        # 计算精确率、召回率和F1分数
        metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # 如果提供了预测概率，计算AUC和AP
        if y_score is not None:
            y_score = np.array(y_score)
            
            # 二分类情况
            if len(np.unique(y_true)) == 2:
                try:
                    metrics["auc"] = float(roc_auc_score(y_true, y_score))
                    metrics["ap"] = float(average_precision_score(y_true, y_score))
                except Exception:
                    pass
            # 多分类情况
            else:
                try:
                    # 确保y_score是概率形式
                    if y_score.ndim == 2 and y_score.shape[1] > 1:
                        metrics["auc"] = float(roc_auc_score(y_true, y_score, average=average, multi_class="ovr"))
                except Exception:
                    pass
                    
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        计算回归任务的评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        metrics = {}
        
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 计算均方误差
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        
        # 计算均方根误差
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        
        # 计算平均绝对误差
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        
        # 计算决定系数
        metrics["r2"] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    @staticmethod
    def calculate_detection_metrics(
        pred_boxes: List[List[float]],
        pred_classes: List[int],
        pred_scores: List[float],
        gt_boxes: List[List[float]],
        gt_classes: List[int],
        iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        计算目标检测任务的评估指标
        
        Args:
            pred_boxes: 预测框，格式为[[x1, y1, x2, y2], ...]
            pred_classes: 预测类别
            pred_scores: 预测分数
            gt_boxes: 真实框，格式为[[x1, y1, x2, y2], ...]
            gt_classes: 真实类别
            iou_threshold: IoU阈值
            
        Returns:
            Dict[str, Any]: 评估指标字典
        """
        metrics = {}
        
        # 计算IoU矩阵
        iou_matrix = MetricsCalculator._calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # 计算mAP
        ap_per_class, mean_ap = MetricsCalculator._calculate_map(
            pred_boxes, pred_classes, pred_scores,
            gt_boxes, gt_classes, iou_threshold
        )
        
        metrics["mAP"] = float(mean_ap)
        metrics["AP_per_class"] = {str(cls_id): float(ap) for cls_id, ap in ap_per_class.items()}
        
        return metrics
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        计算两个框的IoU
        
        Args:
            box1: 第一个框，格式为[x1, y1, x2, y2]
            box2: 第二个框，格式为[x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        if x2 < x1 or y2 < y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # 计算IoU
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _calculate_iou_matrix(boxes1: List[List[float]], boxes2: List[List[float]]) -> np.ndarray:
        """
        计算两组框的IoU矩阵
        
        Args:
            boxes1: 第一组框，格式为[[x1, y1, x2, y2], ...]
            boxes2: 第二组框，格式为[[x1, y1, x2, y2], ...]
            
        Returns:
            np.ndarray: IoU矩阵，形状为(len(boxes1), len(boxes2))
        """
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = MetricsCalculator._calculate_iou(box1, box2)
                
        return iou_matrix
    
    @staticmethod
    def _calculate_map(
        pred_boxes: List[List[float]],
        pred_classes: List[int],
        pred_scores: List[float],
        gt_boxes: List[List[float]],
        gt_classes: List[int],
        iou_threshold: float = 0.5
    ) -> Tuple[Dict[int, float], float]:
        """
        计算mAP
        
        Args:
            pred_boxes: 预测框，格式为[[x1, y1, x2, y2], ...]
            pred_classes: 预测类别
            pred_scores: 预测分数
            gt_boxes: 真实框，格式为[[x1, y1, x2, y2], ...]
            gt_classes: 真实类别
            iou_threshold: IoU阈值
            
        Returns:
            Tuple[Dict[int, float], float]: 每个类别的AP和mAP
        """
        # 获取所有类别
        all_classes = set(pred_classes + gt_classes)
        
        # 计算每个类别的AP
        ap_per_class = {}
        
        for cls_id in all_classes:
            # 筛选当前类别的预测和真实框
            cls_pred_indices = [i for i, c in enumerate(pred_classes) if c == cls_id]
            cls_gt_indices = [i for i, c in enumerate(gt_classes) if c == cls_id]
            
            if not cls_pred_indices or not cls_gt_indices:
                ap_per_class[cls_id] = 0.0
                continue
                
            # 获取当前类别的预测框和分数
            cls_pred_boxes = [pred_boxes[i] for i in cls_pred_indices]
            cls_pred_scores = [pred_scores[i] for i in cls_pred_indices]
            
            # 获取当前类别的真实框
            cls_gt_boxes = [gt_boxes[i] for i in cls_gt_indices]
            
            # 按分数降序排序
            sorted_indices = np.argsort(cls_pred_scores)[::-1]
            cls_pred_boxes = [cls_pred_boxes[i] for i in sorted_indices]
            cls_pred_scores = [cls_pred_scores[i] for i in sorted_indices]
            
            # 计算IoU矩阵
            iou_matrix = MetricsCalculator._calculate_iou_matrix(cls_pred_boxes, cls_gt_boxes)
            
            # 计算TP和FP
            tp = np.zeros(len(cls_pred_boxes))
            fp = np.zeros(len(cls_pred_boxes))
            
            # 标记已匹配的真实框
            matched_gt_boxes = set()
            
            for i, pred_box in enumerate(cls_pred_boxes):
                # 找到IoU最大的真实框
                max_iou = 0.0
                max_j = -1
                
                for j, gt_box in enumerate(cls_gt_boxes):
                    if j in matched_gt_boxes:
                        continue
                    
                    iou = iou_matrix[i, j]
                    if iou > max_iou:
                        max_iou = iou
                        max_j = j
                
                # 如果IoU大于阈值，则为TP，否则为FP
                if max_iou >= iou_threshold and max_j >= 0:
                    tp[i] = 1
                    matched_gt_boxes.add(max_j)
                else:
                    fp[i] = 1
            
            # 计算累积TP和FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            # 计算精确率和召回率
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / len(cls_gt_boxes)
            
            # 计算AP
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11
                
            ap_per_class[cls_id] = ap
        
        # 计算mAP
        mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        
        return ap_per_class, mean_ap 