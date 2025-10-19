"""
ECG生物识别工具包 - MobileNet1D模块

提供ECG信号处理、数据增强、评估指标等功能的统一接口
"""

# 信号处理模块
from .filters import (
    bandpass_filter,
    notch_filter, 
    baseline_correction,
    preprocess_ecg_signal,
    resample_signal
)

# 数据增强模块  
from .augment import ECGAugment1D

# 评估指标模块
from .metrics import (
    compute_accuracy,
    compute_precision_recall_f1,
    compute_sensitivity_specificity,
    compute_top_k_accuracy,
    compute_cmc_curve,
    compute_confusion_matrix,
    compute_roc_auc,
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curves
)

# 种子设置模块
from .seed import set_seed, worker_init_fn

__all__ = [
    # 信号处理
    'bandpass_filter',
    'notch_filter', 
    'baseline_correction',
    'preprocess_ecg_signal',
    'resample_signal',
    
    # 数据增强
    'ECGAugment1D',
    
    # 评估指标
    'compute_accuracy',
    'compute_precision_recall_f1', 
    'compute_sensitivity_specificity',
    'compute_top_k_accuracy',
    'compute_cmc_curve',
    'compute_confusion_matrix',
    'compute_roc_auc',
    'compute_metrics',
    'print_metrics',
    'plot_confusion_matrix',
    'plot_roc_curves',
    
    # 种子设置
    'set_seed',
    'worker_init_fn'
]
