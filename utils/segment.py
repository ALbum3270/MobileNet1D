"""
ECG信号切片模块
支持基于R峰对齐的单周期切片和固定窗口切片
"""

import numpy as np


def segment_by_rpeaks(ecg_signal, rpeaks, fs=250, 
                      before_peak=0.2, after_peak=0.4):
    """
    基于R峰对齐的单周期切片
    
    参数:
        ecg_signal: 输入ECG信号 (1D array)
        rpeaks: R峰位置索引数组
        fs: 采样频率 (Hz)
        before_peak: R峰前多少秒 (s)
        after_peak: R峰后多少秒 (s)
    
    返回:
        segments: 切片列表，每个元素是一个周期的信号
        segment_indices: 每个切片对应的R峰索引
    """
    segments = []
    segment_indices = []
    
    # 转换为采样点数
    before_samples = int(before_peak * fs)
    after_samples = int(after_peak * fs)
    
    for idx, rpeak in enumerate(rpeaks):
        # 计算窗口边界
        start = rpeak - before_samples
        end = rpeak + after_samples
        
        # 检查边界是否有效
        if start >= 0 and end <= len(ecg_signal):
            segment = ecg_signal[start:end]
            segments.append(segment)
            segment_indices.append(idx)
    
    return segments, segment_indices


def segment_fixed_window(ecg_signal, fs=250, window_sec=2.0, stride_sec=0.5):
    """
    固定窗口切片（滑动窗口）
    
    参数:
        ecg_signal: 输入ECG信号 (1D array)
        fs: 采样频率 (Hz)
        window_sec: 窗口长度 (秒)
        stride_sec: 步长 (秒)
    
    返回:
        segments: 切片列表
        start_indices: 每个切片的起始索引
    """
    segments = []
    start_indices = []
    
    # 转换为采样点数
    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    
    # 滑动窗口
    start = 0
    while start + window_samples <= len(ecg_signal):
        segment = ecg_signal[start:start + window_samples]
        segments.append(segment)
        start_indices.append(start)
        start += stride_samples
    
    return segments, start_indices


def normalize_segment(segment):
    """
    归一化单个信号片段
    
    参数:
        segment: 输入信号片段
    
    返回:
        normalized: 归一化后的片段
    """
    mean = np.mean(segment)
    std = np.std(segment)
    
    if std < 1e-8:
        return segment - mean
    
    normalized = (segment - mean) / std
    return normalized


def pad_or_truncate(segment, target_length):
    """
    填充或截断信号片段到目标长度
    
    参数:
        segment: 输入信号片段
        target_length: 目标长度
    
    返回:
        processed_segment: 处理后的片段
    """
    current_length = len(segment)
    
    if current_length == target_length:
        return segment
    elif current_length > target_length:
        # 截断：从中间截取
        start = (current_length - target_length) // 2
        return segment[start:start + target_length]
    else:
        # 填充：用零填充
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        return np.pad(segment, (pad_left, pad_right), mode='constant')


def create_segments(ecg_signal, fs=250, method='rpeak', 
                    rpeaks=None, **kwargs):
    """
    统一的切片接口
    
    参数:
        ecg_signal: 输入ECG信号
        fs: 采样频率 (Hz)
        method: 切片方法，'rpeak' 或 'fixed'
        rpeaks: R峰位置（method='rpeak'时需要）
        **kwargs: 其他参数
    
    返回:
        segments: 切片列表
        metadata: 元数据列表（包含索引等信息）
    """
    if method == 'rpeak':
        if rpeaks is None or len(rpeaks) == 0:
            return [], []
        
        before_peak = kwargs.get('before_peak', 0.2)
        after_peak = kwargs.get('after_peak', 0.4)
        
        segments, indices = segment_by_rpeaks(
            ecg_signal, rpeaks, fs, before_peak, after_peak
        )
        
        metadata = [{'method': 'rpeak', 'rpeak_idx': idx} for idx in indices]
        
    elif method == 'fixed':
        window_sec = kwargs.get('window_sec', 2.0)
        stride_sec = kwargs.get('stride_sec', 0.5)
        
        segments, start_indices = segment_fixed_window(
            ecg_signal, fs, window_sec, stride_sec
        )
        
        metadata = [{'method': 'fixed', 'start_idx': idx} for idx in start_indices]
        
    else:
        raise ValueError(f"未知的切片方法: {method}")
    
    return segments, metadata
