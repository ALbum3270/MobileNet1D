"""
评估指标模块 - 用于ECG生物识别任务的各种评估指标

包含分类指标、医学指标、Top-k准确率、CMC曲线等功能
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算准确率"""
    return accuracy_score(y_true, y_pred)


def compute_precision_recall_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'weighted'
) -> Tuple[float, float, float]:
    """计算精确率、召回率和F1分数
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        average: 平均方式 ('macro', 'weighted', 'micro')
        
    Returns:
        (precision, recall, f1): 精确率、召回率、F1分数
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return precision, recall, f1


def compute_sensitivity_specificity(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'macro'
) -> Tuple[float, float]:
    """计算敏感性(召回率)和特异性
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro', 'weighted')
        
    Returns:
        (sensitivity, specificity): 敏感性、特异性
    """
    # 敏感性就是召回率
    sensitivity = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # 计算特异性 - 需要为每个类别单独计算
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    if n_classes == 2:
        # 二分类情况
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # 多分类情况
        specificities = []
        for i in range(n_classes):
            # 对于类别i，计算特异性
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        
        if average == 'macro':
            specificity = np.mean(specificities)
        elif average == 'weighted':
            # 按类别样本数加权
            class_counts = np.bincount(y_true)
            weights = class_counts / len(y_true)
            specificity = np.average(specificities, weights=weights)
        else:
            specificity = np.mean(specificities)
    
    return sensitivity, specificity


def compute_top_k_accuracy(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    k: int = 5
) -> float:
    """计算Top-k准确率
    
    Args:
        y_true: 真实标签 (N,)
        y_prob: 预测概率 (N, num_classes)
        k: Top-k中的k值
        
    Returns:
        top_k_acc: Top-k准确率
    """
    if len(y_prob.shape) == 1:
        # 如果是1D数组，假设是二分类
        return compute_accuracy(y_true, (y_prob > 0.5).astype(int))
    
    # 获取Top-k预测
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    
    # 检查真实标签是否在Top-k中
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def compute_cmc_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    max_rank: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """计算累积匹配特征(CMC)曲线
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        max_rank: 最大排名，默认为类别数
        
    Returns:
        (ranks, cmc): 排名数组和对应的识别率
    """
    if max_rank is None:
        max_rank = y_prob.shape[1]
    
    ranks = np.arange(1, max_rank + 1)
    cmc = np.zeros(max_rank)
    
    for rank in ranks:
        cmc[rank - 1] = compute_top_k_accuracy(y_true, y_prob, k=rank)
    
    return ranks, cmc


def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 归一化方式 ('true', 'pred', 'all', None)
        
    Returns:
        cm: 混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        normalize: 归一化方式
        title: 图标题
        figsize: 图大小
        save_path: 保存路径
        
    Returns:
        fig: matplotlib图形对象
    """
    cm = compute_confusion_matrix(y_true, y_pred, normalize=normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用seaborn绘制热力图
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_roc_auc(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    multi_class: str = 'ovr'
) -> float:
    """计算ROC AUC分数
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        multi_class: 多类别处理方式 ('ovr', 'ovo')
        
    Returns:
        auc_score: AUC分数
    """
    try:
        if len(np.unique(y_true)) == 2:
            # 二分类
            if len(y_prob.shape) == 2:
                y_prob = y_prob[:, 1]  # 使用正类概率
            return roc_auc_score(y_true, y_prob)
        else:
            # 多分类
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        return 0.0


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'ROC Curves',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        class_names: 类别名称
        title: 图标题
        figsize: 图大小
        save_path: 保存路径
        
    Returns:
        fig: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        # 二分类ROC曲线
        if len(y_prob.shape) == 2:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
    else:
        # 多分类ROC曲线 (One-vs-Rest)
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_score = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, linewidth=2,
                    label=f'{class_name} (AUC = {auc_score:.3f})')
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """计算完整的评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (可选)
        class_names: 类别名称
        average: 平均方式
        
    Returns:
        metrics: 包含各种指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = compute_accuracy(y_true, y_pred)
    
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, average)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # 医学指标
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred, average)
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity
    
    # 如果有概率预测，计算更多指标
    if y_prob is not None:
        # AUC
        metrics['auc'] = compute_roc_auc(y_true, y_prob)
        
        # Top-k准确率
        for k in [1, 3, 5]:
            if y_prob.shape[1] >= k:
                metrics[f'top_{k}_accuracy'] = compute_top_k_accuracy(y_true, y_prob, k)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "评估指标") -> None:
    """打印格式化的指标结果
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:.<30} {value:.4f}")
    
    print(f"{'='*50}\n")
