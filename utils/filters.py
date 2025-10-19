"""
ECG信号滤波模块
包含带通滤波、陷波滤波、基线漂移校正等功能
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40.0, fs=250, order=4):
    """
    带通滤波器：去除低频基线漂移和高频噪声
    
    参数:
        ecg_signal: 输入ECG信号 (1D array)
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数
    
    返回:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 防止频率超出范围
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.001, 0.999)
    
    if low >= high:
        raise ValueError(f"低截止频率 ({lowcut}) 必须小于高截止频率 ({highcut})")
    
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, ecg_signal)
    
    return filtered_signal


def notch_filter(ecg_signal, freq=50.0, fs=250, quality_factor=30):
    """
    陷波滤波器：去除工频干扰 (50Hz 或 60Hz)
    
    参数:
        ecg_signal: 输入ECG信号
        freq: 陷波频率 (Hz)，通常为 50 或 60
        fs: 采样频率 (Hz)
        quality_factor: 品质因数，控制陷波带宽
    
    返回:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs
    freq_normalized = freq / nyquist
    
    # 防止频率超出范围
    if freq_normalized >= 1.0:
        return ecg_signal
    
    b, a = iirnotch(freq_normalized, quality_factor)
    filtered_signal = filtfilt(b, a, ecg_signal)
    
    return filtered_signal


def baseline_correction(ecg_signal, fs=250, cutoff=0.5):
    """
    基线漂移校正：使用高通滤波器去除低频漂移
    
    参数:
        ecg_signal: 输入ECG信号
        fs: 采样频率 (Hz)
        cutoff: 截止频率 (Hz)
    
    返回:
        corrected_signal: 校正后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)
    
    b, a = butter(4, normal_cutoff, btype='high')
    corrected_signal = filtfilt(b, a, ecg_signal)
    
    return corrected_signal


def preprocess_ecg_signal(ecg_signal, fs=250, lowcut=0.5, highcut=40.0, 
                          notch_freq=50.0, apply_notch=True):
    """
    完整的ECG预处理流程
    
    参数:
        ecg_signal: 输入ECG信号
        fs: 采样频率 (Hz)
        lowcut: 带通滤波低截止频率 (Hz)
        highcut: 带通滤波高截止频率 (Hz)
        notch_freq: 陷波频率 (Hz)，50 或 60
        apply_notch: 是否应用陷波滤波
    
    返回:
        processed_signal: 预处理后的信号
    """
    # 1. 带通滤波
    processed = bandpass_filter(ecg_signal, lowcut, highcut, fs)
    
    # 2. 陷波滤波 (可选)
    if apply_notch:
        processed = notch_filter(processed, notch_freq, fs)
    
    # 3. 归一化
    processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-8)
    
    return processed


def resample_signal(ecg_signal, original_fs, target_fs=250):
    """
    重采样信号到目标采样率
    
    参数:
        ecg_signal: 输入ECG信号
        original_fs: 原始采样频率 (Hz)
        target_fs: 目标采样频率 (Hz)
    
    返回:
        resampled_signal: 重采样后的信号
    """
    if original_fs == target_fs:
        return ecg_signal
    
    num_samples = int(len(ecg_signal) * target_fs / original_fs)
    resampled_signal = signal.resample(ecg_signal, num_samples)
    
    return resampled_signal
