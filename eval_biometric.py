#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物识别评估脚本 (工程级最终版, 加强稳健性/性能/可复现性)

特性:
- 读取 train.py 训练得到的 checkpoint（含 config 与 label_mapping）
- AMP + non_blocking 高效提取特征向量 (Embeddings)
- 1:1 验证 (Verification)：AUC、EER、FRR@FAR(1e-2/1e-3)
- O(k) 采样正样本对, 避免 O(n^2) 组合爆炸
- SciPy 可选：无 SciPy 环境自动退化为 NumPy 方案
- 结果落盘：metrics.json、scores_pairs.npz、相似度分布图、ROC 曲线图
- 更稳妥的 data_root 解析（优先脚本同级 data/）
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import sys

# --- SciPy 可选导入 ---
try:
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from sklearn.metrics import roc_curve, auc

# 服务器/无显示环境也能出图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast

# 确保可以导入项目内的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ECGDataset
from model import MobileNet1D


@torch.no_grad()
def get_embeddings(model: MobileNet1D, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 AMP 与 non_blocking 高效提取特征向量。
    返回:
        embeddings: (N, D) float32
        subject_ids: (N,) int64
    """
    model.eval()
    all_embeddings: List[np.ndarray] = []
    all_subject_ids: List[np.ndarray] = []
    use_amp = (device.type == "cuda")

    for batch in tqdm(loader, desc="提取特征向量 (Embeddings)"):
        signals = batch["signal"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            emb = model.extract_features(signals)
        all_embeddings.append(emb.float().cpu().numpy())
        all_subject_ids.append(batch["subject_id"].cpu().numpy())

    if not all_embeddings:
        raise ValueError("评估数据集中没有任何样本；请检查 split_csv 是否为空或路径是否正确。")

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_subject_ids, axis=0)


def _sample_pos_pairs(indices: np.ndarray, k: int, rng: np.random.RandomState) -> List[tuple[int, int]]:
    """
    从同一受试者的样本索引中 O(k) 采样正样本对，避免 O(n^2) 组合爆炸。
    """
    n = len(indices)
    if n < 2 or k <= 0:
        return []
    total = n * (n - 1) // 2
    if k >= total:
        # 样本很少时直接全取
        return list(itertools.combinations(indices.tolist(), 2))

    pairs = set()
    while len(pairs) < k:
        i, j = rng.randint(0, n, size=2)
        if i == j:
            continue
        a, b = int(indices[i]), int(indices[j])
        if a > b:
            a, b = b, a
        pairs.add((a, b))
    return list(pairs)


def create_pairs(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    num_pairs_per_subject: int = 40,
    *,
    seed: int = 42,
) -> Tuple[List[tuple[int, int]], List[tuple[int, int]]]:
    """
    从数据集中创建正样本对（同一受试者）和负样本对（不同受试者），每个受试者最多 num_pairs_per_subject 对。
    采用随机采样，复杂度 O(k)。
    """
    rng = np.random.RandomState(seed)
    unique_sids = np.unique(subject_ids)
    sid_map = {sid: np.where(subject_ids == sid)[0] for sid in unique_sids}

    positive_pairs: List[tuple[int, int]] = []
    negative_pairs: List[tuple[int, int]] = []

    for sid in tqdm(unique_sids, desc="创建正/负样本配对"):
        idxs = sid_map[sid]
        # 正样本对
        positive_pairs.extend(_sample_pos_pairs(idxs, num_pairs_per_subject, rng))
        # 负样本对：每个受试者采样至多 num_pairs_per_subject 对
        for _ in range(num_pairs_per_subject):
            if len(unique_sids) < 2 or len(idxs) == 0:
                break
            i1 = int(rng.choice(idxs))
            neg_sid = sid
            # 直到抽到另一个受试者
            while neg_sid == sid:
                neg_sid = int(rng.choice(unique_sids))
            i2 = int(rng.choice(sid_map[neg_sid]))
            negative_pairs.append((i1, i2))

    return positive_pairs, negative_pairs


def calculate_cosine_similarity(pair_indices: List[tuple[int, int]], embeddings: np.ndarray) -> np.ndarray:
    """
    使用向量化操作快速计算所有配对的余弦相似度。
    """
    if len(pair_indices) == 0:
        return np.array([], dtype=np.float32)
    # 先单位化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    E = embeddings / norms
    idx = np.asarray(pair_indices, dtype=np.int64)
    scores = (E[idx[:, 0]] * E[idx[:, 1]]).sum(axis=1)
    return scores.astype(np.float32)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    计算 EER 及其近似阈值。
    若安装了 SciPy，使用 root-finding；否则退化为 |FPR-FNR| 最小点近似。
    返回:
        (eer, eer_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    if _HAS_SCIPY:
        try:
            tpr_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(tpr[0], tpr[-1]))
            eer = float(brentq(lambda x: 1.0 - tpr_interp(x) - x, 0.0, 1.0))
            # 近似取与 eer 最接近的阈值
            i_eer = int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))
            return eer, float(thresholds[i_eer])
        except Exception:
            pass

    # 纯 NumPy 退化：取 |FPR - FNR| 最小处
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[i] + fpr[i]) / 2.0)
    eer_thr = float(thresholds[i])
    return eer, eer_thr


def frr_at_far(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, far_target: float) -> tuple[float, float]:
    """
    线性插值估计指定 FAR 下的 FRR 以及对应阈值。
    返回:
        (frr, threshold)
    """
    # 边界处理
    if far_target <= fpr[0]:
        return float(1 - tpr[0]), float(thresholds[0])
    if far_target >= fpr[-1]:
        return float(1 - tpr[-1]), float(thresholds[-1])

    i = int(np.searchsorted(fpr, far_target, side="left"))
    # 防止除零
    denom = (fpr[i] - fpr[i - 1]) + 1e-12
    w = (far_target - fpr[i - 1]) / denom
    tpr_i = tpr[i - 1] + w * (tpr[i] - tpr[i - 1])
    thr_i = float(thresholds[i - 1] + w * (thresholds[i] - thresholds[i - 1]))
    return float(1 - tpr_i), thr_i


def main() -> None:
    parser = argparse.ArgumentParser(description="生物识别模型评估脚本（工程级最终版）")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径 (.pt)")
    parser.add_argument("--split_csv", type=str, required=True, help="评估用 CSV (例如 test.csv)")
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录 (不填则自动推断)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="评估时的 batch 大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pairs_per_subject", type=int, default=40, help="每受试者的配对上限")
    parser.add_argument("--seed", type=int, default=42, help="评估随机种子")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认与 ckpt 同级）")
    args = parser.parse_args()

    # --- 可复现性 ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"检查点未找到: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    label_mapping = ckpt.get("label_mapping", {}) or {}
    if not label_mapping:
        raise ValueError("检查点中未找到 'label_mapping'。")

    # --- data_root 解析：优先脚本同级 data/，再回退上一级 ---
    script_dir = Path(__file__).resolve().parent
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        cfg_dir = (cfg.get("data", {}) or {}).get("data_dir")
        if cfg_dir:
            p = Path(cfg_dir)
            data_root = p if p.is_absolute() else (script_dir / p).resolve()
        else:
            cand1 = script_dir / "data"
            cand2 = script_dir.parent / "data"
            data_root = cand1 if cand1.exists() else cand2

    split_csv = Path(args.split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"评估 CSV 未找到: {split_csv}")

    # 推断 target_length（用于 padding/cropping）
    target_length = None
    try:
        fs = int(cfg["data"]["fs"])
        window_sec = float(cfg["data"]["window_sec"])
        target_length = int(round(fs * window_sec))
    except Exception:
        print("警告: 无法从 config 推断 target_length，将使用数据集内部默认策略。")

    # --- 构建数据与模型 ---
    dataset = ECGDataset(
        csv_path=split_csv, data_dir=data_root, label_mapping=label_mapping,
        drop_unmapped=False, transform=None, target_length=target_length,
    )
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = len(label_mapping)
    model = MobileNet1D(num_classes=num_classes)
    # 支持直接 state_dict 或完整 dict
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)

    # --- 提取特征 ---
    embeddings, subject_ids = get_embeddings(model, loader, device)
    uniq_sids = np.unique(subject_ids)
    if uniq_sids.size < 2:
        raise ValueError("需要至少 2 个不同受试者才能进行验证评估。")

    # --- 生成配对并计算分数 ---
    pos_idx, neg_idx = create_pairs(
        embeddings,
        subject_ids,
        num_pairs_per_subject=args.pairs_per_subject,
        seed=args.seed,
    )
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("未能创建足够的样本对进行评估；请增加样本量或调大 pairs_per_subject。")

    pos_scores = calculate_cosine_similarity(pos_idx, embeddings)
    neg_scores = calculate_cosine_similarity(neg_idx, embeddings)

    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])

    # --- 指标 ---
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc_score = float(auc(fpr, tpr))
    eer, eer_thr = compute_eer(labels, scores)
    frr_1e2, thr_1e2 = frr_at_far(fpr, tpr, thresholds, 1e-2)
    frr_1e3, thr_1e3 = frr_at_far(fpr, tpr, thresholds, 1e-3)

    # --- 打印结果 ---
    print("\n" + "=" * 52)
    print("      生物识别验证 (Verification) 评估结果")
    print("=" * 52)
    print(f"  AUC (Area Under Curve): {auc_score:.4f}")
    print(f"  EER (Equal Error Rate): {eer:.4f}  (越低越好)")
    print(f"  EER 阈值(近似):         {eer_thr:.6f}")
    print(f"  FRR @ FAR=1e-2:        {frr_1e2:.4f}  (阈值={thr_1e2:.6f})")
    print(f"  FRR @ FAR=1e-3:        {frr_1e3:.4f}  (阈值={thr_1e3:.6f})")
    print("=" * 52)

    # --- 结果落盘 ---
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / f"eval_biometric_{split_csv.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "auc": auc_score,
        "eer": eer,
        "eer_threshold": eer_thr,
        "frr_at_far_1e-2": frr_1e2,
        "thr_far_1e-2": thr_1e2,
        "frr_at_far_1e-3": frr_1e3,
        "thr_far_1e-3": thr_1e3,
        "num_pos_pairs": int(len(pos_idx)),
        "num_neg_pairs": int(len(neg_idx)),
        "num_subjects": int(uniq_sids.size),
        "num_samples": int(subject_ids.size),
        "seed": int(args.seed),
    }
    with open(out_dir / "biometric_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    np.savez(out_dir / "scores_pairs.npz",
             pos_scores=pos_scores,
             neg_scores=neg_scores,
             fpr=fpr, tpr=tpr, thresholds=thresholds)

    # 相似度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(pos_scores, color="g", kde=True, stat="density", label="正样本对 (同一受试者)")
    sns.histplot(neg_scores, color="r", kde=True, stat="density", label="负样本对 (不同受试者)")
    plt.title("余弦相似度得分分布")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_distribution.png", dpi=300)
    plt.close()

    # ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=300)
    plt.close()

    print(f"\n✓ 评估完成！结果已保存至: {out_dir}")

if __name__ == "__main__":
    main()
