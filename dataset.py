"""
ECG Dataset for identification task.
Loads preprocessed .npy segments indexed by fold CSVs produced by scripts.
Expected CSV columns: 'filepath' (relative to data_dir), 'subject_id', 'length'.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _pad_or_trim(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or center-trim to the target length.

    Assumes 1D array. Returns a copy with length == target_length.
    """
    if target_length <= 0:
        return signal

    current_length = signal.shape[0]
    if current_length == target_length:
        return signal

    if current_length > target_length:
        # Center crop
        start = (current_length - target_length) // 2
        end = start + target_length
        return signal[start:end]

    # Pad both sides as evenly as possible
    total_pad = target_length - current_length
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return np.pad(signal, (left_pad, right_pad), mode="constant", constant_values=0.0)


def create_label_mapping_from_csv(train_csv_path: str | Path) -> Dict[int, int]:
    """Create a mapping from subject_id to contiguous class indices starting at 0."""
    df = pd.read_csv(train_csv_path)
    unique_subjects = sorted(df["subject_id"].unique().tolist())
    return {int(sid): idx for idx, sid in enumerate(unique_subjects)}


class ECGDataset(Dataset):
    """PyTorch Dataset to load ECG segments for identification.

    Parameters:
        csv_path: Path to a fold CSV (train/val/test) with a 'filepath' column
                  that is relative to data_dir, and a 'subject_id' column.
        data_dir: Root data directory (the parent of 'processed').
        label_mapping: Optional mapping from subject_id -> class index. If None,
                       a mapping will be created from the CSV (sorted subjects).
        transform: Optional callable applied on the 1D numpy array before tensor conversion.
        target_length: If provided, signals are padded/trimmed to this length.
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        label_mapping: Optional[Dict[int, int]] = None,
        drop_unmapped: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.csv_path)

        # Normalize dtypes to ensure proper filtering with provided label_mapping
        if "subject_id" in self.df.columns:
            # Cast to integer to avoid type mismatches (e.g., strings vs ints)
            self.df["subject_id"] = pd.to_numeric(self.df["subject_id"], errors="raise").astype(int)

        if "subject_id" not in self.df.columns or "filepath" not in self.df.columns:
            raise ValueError("CSV must contain 'subject_id' and 'filepath' columns")

        # Create or use provided label mapping
        if label_mapping is None:
            subjects = sorted(self.df["subject_id"].unique().tolist())
            self.label_mapping = {int(s): i for i, s in enumerate(subjects)}
        else:
            self.label_mapping = {int(k): int(v) for k, v in label_mapping.items()}

        # Optionally drop rows whose subject_id is not in provided mapping
        if label_mapping is not None and drop_unmapped:
            self.df = (
                self.df[self.df["subject_id"].isin(self.label_mapping.keys())]
                .reset_index(drop=True)
            )

        self.transform = transform
        self.target_length = int(target_length) if target_length is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        # Support both relative and absolute paths in CSV
        file_path_field = str(row["filepath"])  # keep raw string for diagnostics
        path_in_csv = Path(file_path_field)
        subject_id = int(row["subject_id"])

        if path_in_csv.is_absolute():
            abs_path = path_in_csv
        else:
            abs_path = (self.data_dir / path_in_csv).resolve()

        # Ensure path exists; provide actionable diagnostics
        if not abs_path.exists():
            raise FileNotFoundError(
                "ECGDataset: file not found. "
                f"csv='{self.csv_path}', row_index={index}, "
                f"filepath_field='{file_path_field}', resolved_path='{abs_path}', "
                f"data_dir='{self.data_dir}'. "
                "Ensure CSV 'filepath' is absolute or relative to data_dir."
            )

        signal = np.load(str(abs_path))  # (L,)
        signal = np.asarray(signal, dtype=np.float32)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_length is not None:
            signal = _pad_or_trim(signal, self.target_length)

        # Convert to (C=1, L) tensor
        tensor = torch.from_numpy(signal).unsqueeze(0)  # shape: (1, L)
        
        # Handle cases where subject_id is not in label_mapping
        # (e.g., for biometric evaluation with unseen subjects)
        if subject_id in self.label_mapping:
            label = self.label_mapping[subject_id]
        else:
            label = -1  # Use -1 for unmapped subjects

        return {
            "signal": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
        }


