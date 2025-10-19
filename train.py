#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Identification - Training Script (工程级最终版)

关键特性:
- 仅用训练集(train.csv)创建 label_mapping，杜绝数据泄漏
- 训练期使用“受试者不重叠”的内部验证子集；以 Verification(AUC/EER) 作为早停指标
- 若验证子集无法形成验证对(正/负样本对不足)，自动回退为分类验证损失早停
- AMP、梯度裁剪、Cosine LR、metrics.jsonl 持续记录、best_model.pt 保存

依赖: numpy, pandas, torch, scikit-learn, pyyaml, tqdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import yaml
import sys

# 确保本地模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ECGDataset, create_label_mapping_from_csv
from model import MobileNet1D
from utils.seed import set_seed, worker_init_fn
from utils.augment import ECGAugment1D


# ----------------------
# 配置加载
# ----------------------
def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------
# 评估工具：提特征/配对/指标
# ----------------------
@torch.no_grad()
def _extract_embeddings_subjects(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs: List[np.ndarray] = []
    sids: List[np.ndarray] = []
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            z = model.extract_features(x)
        embs.append(z.float().cpu().numpy())
        sids.append(batch["subject_id"].cpu().numpy())
    if not embs:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(embs, axis=0), np.concatenate(sids, axis=0)


def _sample_pos_pairs(indices: np.ndarray, k: int, rng: np.random.RandomState) -> List[tuple[int, int]]:
    """O(k) 采样正样本对(避免 O(n^2) 组合爆炸)。"""
    n = len(indices)
    if n < 2 or k <= 0:
        return []
    total = n * (n - 1) // 2
    if k >= total:
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


def _create_pairs(
    subject_ids: np.ndarray,
    *,
    num_pairs_per_subject: int,
    rng: np.random.RandomState
) -> Tuple[List[tuple[int, int]], List[tuple[int, int]]]:
    """从 subject_ids 生成正/负样本对的索引对列表。"""
    unique_sids = np.unique(subject_ids)
    sid_map = {sid: np.where(subject_ids == sid)[0] for sid in unique_sids}
    pos_pairs: List[tuple[int, int]] = []
    neg_pairs: List[tuple[int, int]] = []
    for sid in unique_sids:
        idxs = sid_map[sid]
        # 正样本对
        pos_pairs.extend(_sample_pos_pairs(idxs, num_pairs_per_subject, rng))
        # 负样本对
        for _ in range(num_pairs_per_subject):
            if unique_sids.size < 2 or idxs.size == 0:
                break
            i1 = int(rng.choice(idxs))
            neg_sid = sid
            while neg_sid == sid:
                neg_sid = int(rng.choice(unique_sids))
            i2 = int(rng.choice(sid_map[neg_sid]))
            neg_pairs.append((i1, i2))
    return pos_pairs, neg_pairs


def _cosine_scores(pair_indices: List[tuple[int, int]], embeddings: np.ndarray) -> np.ndarray:
    """向量化计算指定配对的余弦相似度。"""
    if len(pair_indices) == 0:
        return np.zeros((0,), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    E = embeddings / norms
    idx = np.asarray(pair_indices, dtype=np.int64)
    return (E[idx[:, 0]] * E[idx[:, 1]]).sum(axis=1).astype(np.float32)


def _eer_from_fpr_tpr(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """不依赖 SciPy 的 EER 近似：取 |FPR - FNR| 最小点。"""
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2.0)


@torch.no_grad()
def verification_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    num_pairs_per_subject: int,
    random_seed: int
) -> Tuple[float | float("nan"), float | float("nan")]:
    """
    计算 Verification 指标 (AUC, EER)。
    如果验证集无法形成正/负样本对，返回 (nan, nan)。
    """
    embs, sids = _extract_embeddings_subjects(model, loader, device, use_amp=use_amp)
    uniq = np.unique(sids)
    if embs.shape[0] == 0 or uniq.size < 2:
        return float("nan"), float("nan")

    rng = np.random.RandomState(random_seed)
    pos_idx, neg_idx = _create_pairs(sids, num_pairs_per_subject=num_pairs_per_subject, rng=rng)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan"), float("nan")

    pos_scores = _cosine_scores(pos_idx, embs)
    neg_scores = _cosine_scores(neg_idx, embs)

    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])

    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return float(auc(fpr, tpr)), _eer_from_fpr_tpr(fpr, tpr)


# ----------------------
# 训练/评估
# ----------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: GradScaler,
    grad_clip_norm: float
) -> Tuple[float, float]:
    model.train()
    total, loss_sum, correct = 0, 0.0, 0.0
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        bsz = y.size(0)
        total += bsz
        loss_sum += loss.item() * bsz
        correct += (logits.argmax(1) == y).float().sum().item()
    return (loss_sum / total if total else float("nan"),
            correct / total if total else float("nan"))


@torch.no_grad()
def evaluate_cls(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool
) -> Tuple[float, float]:
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0.0
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        bsz = y.size(0)
        total += bsz
        loss_sum += loss.item() * bsz
        correct += (logits.argmax(1) == y).float().sum().item()
    return (loss_sum / total if total else float("nan"),
            correct / total if total else float("nan"))


# ----------------------
# 主流程
# ----------------------
def main() -> None:
    import itertools  # 延迟导入供 _sample_pos_pairs 使用

    parser = argparse.ArgumentParser(description="Train 1D CNN for ECG identification (工程级最终版)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--processed_dir", type=str, default=None, help="包含 train/val/test.csv 的目录(不填则从 config 推断)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="可选: 预训练/继续训练的 checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("split", {}).get("random_seed", 42))
    set_seed(seed)

    # --- 数据路径解析 ---
    project_root = Path(__file__).resolve().parent
    data_dir_cfg = str(cfg.get("data", {}).get("data_dir", "")).strip()
    if data_dir_cfg:
        p = Path(data_dir_cfg)
        data_root = p if p.is_absolute() else (project_root / p).resolve()
    else:
        cand1 = project_root / "data"
        cand2 = project_root.parent / "data"
        data_root = cand1 if cand1.exists() else cand2

    dataset_name = cfg["data"]["dataset"]
    lead = cfg["data"]["lead"]
    slice_method = cfg["data"]["slice_method"]
    processed_dir = Path(args.processed_dir) if args.processed_dir else (data_root / "processed" / dataset_name / f"{slice_method}_{lead}")

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"未找到训练集: {train_csv} (请先运行划分脚本)")

    # --- 仅使用训练集构建 label_mapping ---
    label_mapping = create_label_mapping_from_csv(train_csv)
    num_classes = len(label_mapping)
    print(f"✓ 仅用训练集构建标签映射，共 {num_classes} 类。")

    # --- 推断 target_length ---
    target_length: Optional[int] = None
    try:
        df_train_len = pd.read_csv(train_csv)
        if "length" in df_train_len.columns and not df_train_len["length"].isnull().all():
            target_length = int(float(df_train_len["length"].median()))
    except Exception:
        target_length = None
    if target_length is None:
        try:
            fs = int(cfg["data"]["fs"])
            window_sec = float(cfg["data"]["window_sec"])
            target_length = int(round(fs * window_sec))
        except Exception:
            target_length = None

    # --- 构建 full_train_set (无增强) ---
    full_train_set = ECGDataset(
        train_csv, data_dir=data_root, label_mapping=label_mapping,
        drop_unmapped=True, transform=None, target_length=target_length
    )

    # --- 受试者不重叠的内部划分 ---
    val_ratio = float(cfg.get("training", {}).get("within_subject_val_ratio", 0.15))
    df_train = full_train_set.df
    unique_subjects = df_train["subject_id"].unique()
    rng = np.random.RandomState(seed)
    # 尝试让验证受试者具备形成验证对的能力
    sid_counts = df_train["subject_id"].value_counts()
    candidates = set(sid_counts[sid_counts >= 2].index.tolist())

    def sample_val_subjects(all_sids: List[int], wish_k: int) -> set:
        all_sids = list(all_sids)
        rng.shuffle(all_sids)
        # 优先取 candidates，再补非 candidates
        cand = [s for s in all_sids if s in candidates]
        non_cand = [s for s in all_sids if s not in candidates]
        pick = (cand + non_cand)[:max(2, wish_k)]
        return set(pick)

    if unique_subjects.size >= 2:
        wish_k = max(2, int(round(unique_subjects.size * val_ratio)))
        val_subjects = sample_val_subjects(unique_subjects.tolist(), wish_k)
        train_subjects = set(unique_subjects.tolist()) - val_subjects
    else:
        val_subjects = set()
        train_subjects = set(unique_subjects.tolist())

    train_idx = df_train.index[df_train["subject_id"].isin(train_subjects)].to_list()
    val_idx = df_train.index[df_train["subject_id"].isin(val_subjects)].to_list()

    # 条件检查：验证集中是否有 >=1 位受试者样本数>=2 且 受试者数>=2
    val_sid_counts = df_train.loc[val_idx, "subject_id"].value_counts() if len(val_idx) else pd.Series(dtype=int)
    cond_pos = (val_sid_counts >= 2).any()
    cond_neg = (val_sid_counts.index.nunique() >= 2)
    use_verification_earlystop = bool(cond_pos and cond_neg)
    if not use_verification_earlystop:
        print("⚠️ 验证集暂不满足形成验证对的条件，将回退到分类验证损失作为早停指标。")

    # --- 数据增强(仅训练子集) ---
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("apply_prob", 0) > 0:
        fs = int(cfg["data"]["fs"])
        train_transform = ECGAugment1D(fs=fs, **aug_cfg)
        train_set_aug = ECGDataset(train_csv, data_dir=data_root, label_mapping=label_mapping,
                                   drop_unmapped=True, transform=train_transform, target_length=target_length)
        train_subset = Subset(train_set_aug, train_idx)
        val_subset = Subset(full_train_set, val_idx)
    else:
        train_subset = Subset(full_train_set, train_idx)
        val_subset = Subset(full_train_set, val_idx)

    # --- Test loader (用于信息提示/可能的分类评估) ---
    test_set = ECGDataset(test_csv, data_dir=data_root, label_mapping=label_mapping,
                          drop_unmapped=True, transform=None, target_length=target_length)

    # --- DataLoaders ---
    trn = cfg["training"]
    batch_size = int(trn["batch_size"])
    num_workers = int(trn["num_workers"])
    pin_memory = bool(trn.get("pin_memory", True))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"✓ 内部划分: 训练受试者 {len(train_subjects)} | 验证受试者 {len(val_subjects)}")
    print(f"  样本数: 训练 {len(train_subset)} | 验证 {len(val_subset)}")
    if len(test_loader.dataset) == 0:
        print("  ⚠️ 跨受试者测试集为空（受试者与训练不相交），最终评估请使用 eval_biometric.py 做 1:1 验证。")

    # --- 模型/优化器/调度器 ---
    device = torch.device(args.device)
    model = MobileNet1D(num_classes=num_classes).to(device)
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state.get("model_state", state), strict=False)
            print(f"✓ 已加载预训练/继续训练权重: {ckpt_path}")

    opt_cfg = trn["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"])
    )
    epochs = int(trn["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(trn.get("scheduler", {}).get("eta_min", 1e-5))
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(trn.get("label_smoothing", 0.0)))
    use_amp = bool(trn.get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    grad_clip_norm = float(trn.get("grad_clip_norm", 0.0))

    # --- 保存目录/日志 ---
    model_name = "mobilenet1d"
    save_root_cfg = Path(cfg["logging"]["save_dir"])
    save_root = save_root_cfg if save_root_cfg.is_absolute() else (project_root / save_root_cfg).resolve()
    save_dir = save_root / dataset_name / f"{slice_method}_{lead}" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "metrics.jsonl"
    ckpt_path = save_dir / "best_model.pt"

    ver_metric_name = str(trn.get("verification_metric", "auc")).lower()
    ver_pairs_per_subject = int(trn.get("verification_num_pairs_per_subject", 40))
    if ver_metric_name not in {"auc", "eer"}:
        ver_metric_name = "auc"

    # 早停状态
    if use_verification_earlystop:
        best_metric = float("-inf") if ver_metric_name == "auc" else float("inf")
    else:
        # 回退为分类验证损失，目标最小
        best_metric = float("-inf")  # 我们用 current_metric = -val_loss 来“越大越好”
    patience = int(trn.get("early_stopping_patience", 10))
    patience_cnt = 0

    # --- 训练循环 ---
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=use_amp, scaler=scaler, grad_clip_norm=grad_clip_norm
        )
        val_loss, val_acc = evaluate_cls(model, val_loader, criterion, device, use_amp=use_amp)

        # 验证的 Verification 指标（若可用）
        val_auc, val_eer = float("nan"), float("nan")
        if use_verification_earlystop:
            val_auc, val_eer = verification_on_loader(
                model, val_loader, device, use_amp=use_amp,
                num_pairs_per_subject=ver_pairs_per_subject,
                random_seed=seed,
            )

        # 记录日志
        def _nan_to_none(x):
            return None if (isinstance(x, float) and np.isnan(x)) else x

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_cls_loss": val_loss,
            "val_cls_acc": val_acc,
            "val_auc": _nan_to_none(val_auc),
            "val_eer": _nan_to_none(val_eer),
            "lr": optimizer.param_groups[0]["lr"],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 控制台输出
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val(clf) Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"Val(verify) AUC {val_auc:.4f} EER {val_eer:.4f}"
        )

        # 早停指标
        if use_verification_earlystop:
            cur = val_auc if ver_metric_name == "auc" else val_eer
            is_better = (cur > best_metric) if ver_metric_name == "auc" else (cur < best_metric)
        else:
            # 回退: 按 -val_loss 最大化
            cur = -val_loss
            is_better = cur > best_metric

        if not (isinstance(cur, float) and np.isnan(cur)) and is_better:
            best_metric = cur
            patience_cnt = 0
            torch.save({"model_state": model.state_dict(),
                        "label_mapping": label_mapping,
                        "config": cfg}, ckpt_path)
            tag = "AUC" if use_verification_earlystop and ver_metric_name == "auc" else ("EER" if use_verification_earlystop else "VAL-CLS-LOSS")
            val_show = cur if tag != "EER" else val_eer
            print(f"✓ 保存新最佳模型 -> {ckpt_path} (Best {tag}: {val_show:.6f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("触发提前停止。")
                break

        scheduler.step()

    # --- 最终测试（仅在 test_loader 非空时做分类评估；cross-subject 建议用 eval_biometric.py） ---
    print("\n在测试集上进行最终分类评估...")
    if len(test_loader.dataset) == 0:
        print("--> 跳过：cross-subject 测试集与训练标签不相交。请使用 eval_biometric.py 做 1:1 验证。")
    elif ckpt_path.exists():
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["model_state"])
        test_loss, test_acc = evaluate_cls(model, test_loader, criterion, device, use_amp=use_amp)
        print(f"--> 测试集(分类)结果: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_test_loss": test_loss, "final_test_acc": test_acc}, ensure_ascii=False) + "\n")
    else:
        print("未找到最佳模型检查点，跳过测试评估。")


if __name__ == "__main__":
    main()
