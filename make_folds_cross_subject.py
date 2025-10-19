#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-subject split: ensure subject_id sets are disjoint across train/val/test.
Usage:
    python make_folds_cross_subject.py --samples_csv /path/to/samples.csv --out_dir ./splits
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="执行跨受试者（Subject-Disjoint）的数据集划分。")
    ap.add_argument("--samples_csv", required=True, help="包含所有样本信息的总CSV文件路径")
    ap.add_argument("--out_dir", required=True, help="保存 train.csv, val.csv, test.csv 的输出目录")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="训练集受试者比例")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="验证集受试者比例")
    ap.add_argument("--test_ratio", type=float, default=0.15, help="测试集受试者比例")
    ap.add_argument("--seed", type=int, default=42, help="用于复现划分的随机种子")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.samples_csv)
    if "subject_id" not in df.columns:
        raise KeyError("列 'subject_id' 在 samples_csv 中未找到")

    subjects = sorted(df["subject_id"].astype(int).unique().tolist())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * args.train_ratio))
    n_val   = int(round(n * args.val_ratio))
    train_ids = set(subjects[:n_train])
    val_ids   = set(subjects[n_train:n_train+n_val])
    test_ids  = set(subjects[n_train+n_val:])

    def pick(ids):
        return df[df["subject_id"].astype(int).isin(ids)].reset_index(drop=True)

    train_df, val_df, test_df = pick(train_ids), pick(val_ids), pick(test_ids)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids), \
        "错误：划分后的集合存在受试者重叠！"
    print(f"✓ 划分完成。受试者数量: 训练集={len(train_ids)}, 验证集={len(val_ids)}, 测试集={len(test_ids)}")
    print(f"   样本数量: 训练集={len(train_df)}, 验证集={len(val_df)}, 测试集={len(test_df)}")
    print(f"   文件已保存至: {out.resolve()}")

if __name__ == "__main__":
    main()