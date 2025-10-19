"""
R峰检测模块
使用NeuroKit2库进行R峰检测，并提供质量验证功能
"""

import numpy as np
import neurokit2 as nk


def detect_rpeaks(ecg_signal, fs=250, method='neurokit'):
    """
    检测ECG信号中的R峰位置
    
    参数:
        ecg_signal: 输入ECG信号 (1D array)
        fs: 采样频率 (Hz)
        method: 检测方法，可选 'neurokit', 'pantompkins', 'hamilton'
    
    返回:
        rpeaks: R峰位置索引数组
    """
    try:
        # 使用NeuroKit2进行R峰检测
        _, rpeaks_dict = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method=method)
        rpeaks = rpeaks_dict['ECG_R_Peaks']
        
        return rpeaks
    
    except Exception as e:
        print(f"R峰检测失败: {e}")
        return np.array([])


def validate_rpeaks(rpeaks, fs=250, min_rr_ms=300, max_rr_ms=2000):
    """
    验证R峰质量：检查RR间期是否在合理范围内
    
    参数:
        rpeaks: R峰位置索引数组
        fs: 采样频率 (Hz)
        min_rr_ms: 最小RR间期 (毫秒)，对应最大心率 200 bpm
        max_rr_ms: 最大RR间期 (毫秒)，对应最小心率 30 bpm
    
    返回:
        valid_rpeaks: 过滤后的有效R峰数组
        quality_score: 质量分数 (0-1)
    """
    if len(rpeaks) < 2:
        return rpeaks, 0.0
    
    # 计算RR间期 (采样点)
    rr_intervals = np.diff(rpeaks)
    
    # 转换为毫秒
    rr_intervals_ms = (rr_intervals / fs) * 1000
    
    # 定义合理范围
    min_rr_samples = (min_rr_ms / 1000) * fs
    max_rr_samples = (max_rr_ms / 1000) * fs
    
    # 找出有效的RR间期
    valid_mask = (rr_intervals >= min_rr_samples) & (rr_intervals <= max_rr_samples)
    
    # 构建有效R峰数组（保留第一个R峰 + 有效间期对应的后续R峰）
    valid_indices = np.concatenate([[True], valid_mask])
    valid_rpeaks = rpeaks[valid_indices]
    
    # 计算质量分数
    quality_score = np.sum(valid_mask) / len(valid_mask) if len(valid_mask) > 0 else 0.0
    
    return valid_rpeaks, quality_score


def compute_heart_rate(rpeaks, fs=250):
    """
    计算平均心率
    
    参数:
        rpeaks: R峰位置索引数组
        fs: 采样频率 (Hz)
    
    返回:
        heart_rate: 平均心率 (bpm)
    """
    if len(rpeaks) < 2:
        return None
    
    rr_intervals = np.diff(rpeaks) / fs  # 转换为秒
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60.0 / mean_rr
    
    return heart_rate


def detect_and_validate_rpeaks(ecg_signal, fs=250, method='neurokit',
                                 min_rr_ms=300, max_rr_ms=2000,
                                 min_quality=0.7):
    """
    检测并验证R峰的完整流程
    
    参数:
        ecg_signal: 输入ECG信号
        fs: 采样频率 (Hz)
        method: R峰检测方法
        min_rr_ms: 最小RR间期 (毫秒)
        max_rr_ms: 最大RR间期 (毫秒)
        min_quality: 最小质量分数阈值
    
    返回:
        rpeaks: 有效R峰位置
        quality_score: 质量分数
        is_valid: 是否通过质量检查
    """
    # 检测R峰
    rpeaks = detect_rpeaks(ecg_signal, fs, method)
    
    if len(rpeaks) < 2:
        return rpeaks, 0.0, False
    
    # 验证R峰质量
    valid_rpeaks, quality_score = validate_rpeaks(rpeaks, fs, min_rr_ms, max_rr_ms)
    
    # 判断是否通过质量检查
    is_valid = quality_score >= min_quality and len(valid_rpeaks) >= 2
    
    return valid_rpeaks, quality_score, is_valid
