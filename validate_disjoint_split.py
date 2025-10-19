#!/usr/bin/env python3
"""
验证Subject-Disjoint数据划分的正确性
"""

import pandas as pd
import json
from pathlib import Path
import argparse

def validate_split(split_dir):
    """验证数据划分的正确性"""
    split_path = Path(split_dir)
    
    print("=" * 60)
    print("Subject-Disjoint 数据划分验证")
    print("=" * 60)
    
    # 读取数据文件
    train_df = pd.read_csv(split_path / 'train.csv')
    val_df = pd.read_csv(split_path / 'val.csv')
    test_df = pd.read_csv(split_path / 'test.csv')
    
    # 读取划分信息
    with open(split_path / 'split_info.json', 'r') as f:
        split_info = json.load(f)
    
    print(f"数据集统计:")
    print(f"  训练集: {len(train_df)} 样本, {train_df['subject_id'].nunique()} 受试者")
    print(f"  验证集: {len(val_df)} 样本, {val_df['subject_id'].nunique()} 受试者")
    print(f"  测试集: {len(test_df)} 样本, {test_df['subject_id'].nunique()} 受试者")
    print(f"  总计: {len(train_df) + len(val_df) + len(test_df)} 样本")
    
    # 获取受试者集合
    train_subjects = set(train_df['subject_id'].unique())
    val_subjects = set(val_df['subject_id'].unique())
    test_subjects = set(test_df['subject_id'].unique())
    
    print(f"\n受试者分布:")
    print(f"  训练集受试者: {sorted(train_subjects)}")
    print(f"  验证集受试者: {sorted(val_subjects)}")
    print(f"  测试集受试者: {sorted(test_subjects)}")
    
    # 验证无重叠
    print(f"\n重叠检查:")
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    print(f"  训练集 ∩ 验证集: {len(train_val_overlap)} 个受试者 {list(train_val_overlap) if train_val_overlap else '(无重叠)'}")
    print(f"  训练集 ∩ 测试集: {len(train_test_overlap)} 个受试者 {list(train_test_overlap) if train_test_overlap else '(无重叠)'}")
    print(f"  验证集 ∩ 测试集: {len(val_test_overlap)} 个受试者 {list(val_test_overlap) if val_test_overlap else '(无重叠)'}")
    
    # 验证完整性
    all_subjects = train_subjects | val_subjects | test_subjects
    total_subjects = len(train_subjects) + len(val_subjects) + len(test_subjects)
    
    print(f"\n完整性检查:")
    print(f"  所有受试者数: {len(all_subjects)}")
    print(f"  各集合受试者数之和: {total_subjects}")
    print(f"  是否完整且无重叠: {'✓' if len(all_subjects) == total_subjects else '✗'}")
    
    # 验证与split_info的一致性
    print(f"\n与split_info.json的一致性:")
    info_train_subjects = set(split_info['train_subjects'])
    info_val_subjects = set(split_info['val_subjects'])
    info_test_subjects = set(split_info['test_subjects'])
    
    train_match = train_subjects == info_train_subjects
    val_match = val_subjects == info_val_subjects
    test_match = test_subjects == info_test_subjects
    
    print(f"  训练集受试者匹配: {'✓' if train_match else '✗'}")
    print(f"  验证集受试者匹配: {'✓' if val_match else '✗'}")
    print(f"  测试集受试者匹配: {'✓' if test_match else '✗'}")
    
    # 样本数量验证
    print(f"\n样本数量验证:")
    print(f"  训练集样本数 (实际/记录): {len(train_df)}/{split_info['train_samples']} {'✓' if len(train_df) == split_info['train_samples'] else '✗'}")
    print(f"  验证集样本数 (实际/记录): {len(val_df)}/{split_info['val_samples']} {'✓' if len(val_df) == split_info['val_samples'] else '✗'}")
    print(f"  测试集样本数 (实际/记录): {len(test_df)}/{split_info['test_samples']} {'✓' if len(test_df) == split_info['test_samples'] else '✗'}")
    
    # 总结
    all_checks_pass = (
        len(train_val_overlap) == 0 and
        len(train_test_overlap) == 0 and
        len(val_test_overlap) == 0 and
        len(all_subjects) == total_subjects and
        train_match and val_match and test_match and
        len(train_df) == split_info['train_samples'] and
        len(val_df) == split_info['val_samples'] and
        len(test_df) == split_info['test_samples']
    )
    
    print(f"\n" + "=" * 60)
    if all_checks_pass:
        print("✅ 所有验证通过！数据划分正确且符合Subject-Disjoint要求。")
    else:
        print("❌ 验证失败！数据划分存在问题。")
    print("=" * 60)
    
    return all_checks_pass

def main():
    parser = argparse.ArgumentParser(description='验证Subject-Disjoint数据划分')
    parser.add_argument('split_dir', type=str, help='数据划分目录路径')
    args = parser.parse_args()
    
    validate_split(args.split_dir)

if __name__ == '__main__':
    main()
