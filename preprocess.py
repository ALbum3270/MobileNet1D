#!/usr/bin/env python3
"""
主预处理脚本
完整流程：读取原始数据 → 预处理 → R峰检测 → 切片 → 保存
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
import json

# 添加当前目录到路径以导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from utils.filters import preprocess_ecg_signal, resample_signal
from utils.rpeaks import detect_and_validate_rpeaks
from utils.segment import create_segments, normalize_segment


def load_ptbxl_record(record_path, lead='II', target_fs=250):
    """
    加载PTB-XL数据集的一条记录
    
    参数:
        record_path: 记录路径（不含扩展名）
        lead: 导联名称
        target_fs: 目标采样频率
    
    返回:
        signal: ECG信号
        fs: 采样频率
    """
    try:
        record = wfdb.rdrecord(str(record_path))
        
        # PTB-XL的导联顺序
        lead_names = record.sig_name
        if lead not in lead_names:
            return None, None
        
        lead_idx = lead_names.index(lead)
        signal = record.p_signal[:, lead_idx]
        original_fs = record.fs
        
        # 重采样
        if original_fs != target_fs:
            signal = resample_signal(signal, original_fs, target_fs)
        
        return signal, target_fs
        
    except Exception as e:
        print(f"加载失败 {record_path}: {e}")
        return None, None


def load_ecgid_record(record_path, lead='I', target_fs=250):
    """
    加载ECG-ID数据集的一条记录
    
    参数:
        record_path: 记录路径（不含扩展名）
        lead: 导联名称（ECG-ID只有Lead I和Lead II）
        target_fs: 目标采样频率
    
    返回:
        signal: ECG信号
        fs: 采样频率
    """
    try:
        record = wfdb.rdrecord(str(record_path))
        
        # ECG-ID的导联
        lead_names = record.sig_name
        if lead not in lead_names:
            # ECG-ID可能用'ECG I', 'ECG II'命名
            alt_lead = f'ECG {lead}'
            if alt_lead in lead_names:
                lead = alt_lead
            else:
                return None, None
        
        lead_idx = lead_names.index(lead)
        signal = record.p_signal[:, lead_idx]
        original_fs = record.fs
        
        # 重采样
        if original_fs != target_fs:
            signal = resample_signal(signal, original_fs, target_fs)
        
        return signal, target_fs
        
    except Exception as e:
        print(f"加载失败 {record_path}: {e}")
        return None, None


def process_dataset(dataset_name, data_dir, lead='II', target_fs=250,
                    slice_method='fixed', output_dir=None,
                    max_subjects=None, max_records=None):
    """
    处理整个数据集
    
    参数:
        dataset_name: 数据集名称 ('ptbxl' 或 'ecgid')
        data_dir: 数据目录
        lead: 导联名称
        target_fs: 目标采样频率
        slice_method: 切片方法 ('rpeak' 或 'fixed')
        output_dir: 输出目录
    
    返回:
        samples_df: 包含所有样本信息的DataFrame
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / 'raw' / dataset_name
    
    if output_dir is None:
        output_dir = data_dir / 'processed' / dataset_name / f'{slice_method}_{lead}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name.upper()}")
    print(f"导联: {lead}")
    print(f"切片方法: {slice_method}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    samples = []
    
    if dataset_name == 'ptbxl':
        # PTB-XL数据集处理
        samples = process_ptbxl(
            raw_dir, lead, target_fs, slice_method, output_dir, data_dir,
            max_subjects=max_subjects, max_records=max_records
        )
    
    elif dataset_name == 'ecgid':
        # ECG-ID数据集处理
        samples = process_ecgid(raw_dir, lead, target_fs, slice_method, output_dir, data_dir)
    
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
    
    # 转换为DataFrame
    samples_df = pd.DataFrame(samples)
    
    # 保存索引文件
    index_path = output_dir / 'samples.csv'
    samples_df.to_csv(index_path, index=False)
    
    print(f"\n✓ 处理完成！")
    print(f"  总样本数: {len(samples_df)}")
    if len(samples_df) > 0:
        print(f"  总受试者数: {samples_df['subject_id'].nunique()}")
        print(f"  索引文件: {index_path}")
    
    return samples_df


def process_ptbxl(raw_dir, lead, target_fs, slice_method, output_dir, data_dir,
                  max_subjects=None, max_records=None):
    """处理PTB-XL数据集"""
    samples = []
    
    # 读取元数据
    database_path = raw_dir / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    
    if not database_path.exists():
        print(f"错误: PTB-XL数据集未找到: {database_path}")
        print("请先运行: python preprocess.py")
        return samples
    
    metadata_file = database_path / 'ptbxl_database.csv'
    if not metadata_file.exists():
        print(f"错误: 元数据文件未找到: {metadata_file}")
        return samples
    
    metadata = pd.read_csv(metadata_file)

    # 可选：限制受试者数量或记录数量（用于快速预处理）
    if max_subjects is not None:
        try:
            subjects = sorted(pd.unique(metadata['patient_id']).tolist())
            keep_subjects = set(subjects[:int(max_subjects)])
            metadata = metadata[metadata['patient_id'].isin(keep_subjects)].copy()
        except Exception:
            pass
    if max_records is not None:
        metadata = metadata.iloc[:int(max_records)].copy()

    print(f"找到 {len(metadata)} 条记录")
    
    sample_id = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="处理PTB-XL"):
        # 获取记录路径
        filename_lr = row['filename_lr']  # 使用低采样率版本(100Hz)
        record_path = database_path / filename_lr.replace('.dat', '')
        
        subject_id = int(row['patient_id'])
        
        # 加载信号
        signal, fs = load_ptbxl_record(record_path, lead, target_fs)
        
        if signal is None:
            continue
        
        # 预处理
        try:
            clean_signal = preprocess_ecg_signal(signal, fs)
        except Exception as e:
            continue
        
        # R峰检测（如果需要）
        rpeaks = None
        if slice_method == 'rpeak':
            rpeaks, quality, is_valid = detect_and_validate_rpeaks(clean_signal, fs)
            if not is_valid:
                # R峰质量不佳，跳过或回退到固定窗
                continue
        
        # 切片
        segments, metadata_list = create_segments(
            clean_signal, fs, method=slice_method, rpeaks=rpeaks
        )
        
        # 保存每个切片
        for seg_idx, (segment, seg_meta) in enumerate(zip(segments, metadata_list)):
            # 归一化
            segment_norm = normalize_segment(segment)
            
            # 保存为.npy文件
            filename = f'sample_{sample_id:06d}.npy'
            filepath = output_dir / filename
            np.save(filepath, segment_norm.astype(np.float32))
            
            # 记录样本信息
            samples.append({
                'sample_id': sample_id,
                'subject_id': subject_id,
                'dataset': 'ptbxl',
                'lead': lead,
                'fs': fs,
                'slice_method': slice_method,
                'filepath': str(filepath.relative_to(data_dir)),
                'length': len(segment_norm),
                'segment_idx': seg_idx,
            })
            
            sample_id += 1
    
    return samples


def process_ecgid(raw_dir, lead, target_fs, slice_method, output_dir, data_dir):
    """处理ECG-ID数据集"""
    samples = []
    sample_id = 0
    
    # ECG-ID有90个受试者
    person_dirs = sorted(raw_dir.glob('Person_*'))
    
    if len(person_dirs) == 0:
        print(f"错误: ECG-ID数据集未找到: {raw_dir}")
        print("请先下载ECG-ID数据集")
        return samples
    
    print(f"找到 {len(person_dirs)} 个受试者")
    
    for person_dir in tqdm(person_dirs, desc="处理ECG-ID"):
        subject_id = int(person_dir.name.split('_')[1])
        
        # 每个受试者有2条记录
        for rec_num in [1, 2]:
            record_path = person_dir / f'rec_{rec_num}'
            
            # 加载信号
            signal, fs = load_ecgid_record(record_path, lead, target_fs)
            
            if signal is None:
                continue
            
            # 预处理
            try:
                clean_signal = preprocess_ecg_signal(signal, fs)
            except Exception as e:
                continue
            
            # R峰检测（如果需要）
            rpeaks = None
            if slice_method == 'rpeak':
                rpeaks, quality, is_valid = detect_and_validate_rpeaks(clean_signal, fs)
                if not is_valid:
                    continue
            
            # 切片
            segments, metadata_list = create_segments(
                clean_signal, fs, method=slice_method, rpeaks=rpeaks
            )
            
            # 保存每个切片
            for seg_idx, (segment, seg_meta) in enumerate(zip(segments, metadata_list)):
                # 归一化
                segment_norm = normalize_segment(segment)
                
                # 保存为.npy文件
                filename = f'sample_{sample_id:06d}.npy'
                filepath = output_dir / filename
                np.save(filepath, segment_norm.astype(np.float32))
                
                # 记录样本信息
                samples.append({
                    'sample_id': sample_id,
                    'subject_id': subject_id,
                    'dataset': 'ecgid',
                    'lead': lead,
                    'fs': fs,
                    'slice_method': slice_method,
                    'filepath': str(filepath.relative_to(data_dir)),
                    'length': len(segment_norm),
                    'segment_idx': seg_idx,
                    'session_id': rec_num,
                })
                
                sample_id += 1
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='ECG数据预处理')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ptbxl', 'ecgid'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--lead', type=str, default='II',
                        help='导联名称 (默认: II)')
    parser.add_argument('--fs', type=int, default=250,
                        help='目标采样频率 (默认: 250 Hz)')
    parser.add_argument('--slice', type=str, default='fixed',
                        choices=['rpeak', 'fixed'],
                        help='切片方法 (默认: fixed)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: data/processed/{dataset}/{slice}_{lead})')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='仅处理前 N 个受试者（用于快速调试）')
    parser.add_argument('--max_records', type=int, default=None,
                        help='仅处理前 N 条记录（用于快速调试）')
    
    args = parser.parse_args()
    
    # 处理数据集
    samples_df = process_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        lead=args.lead,
        target_fs=args.fs,
        slice_method=args.slice,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        max_records=args.max_records
    )
    
    print(f"\n样本统计:")
    print(samples_df.groupby('subject_id').size().describe())


if __name__ == '__main__':
    main()
