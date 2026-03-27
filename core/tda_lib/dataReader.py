'''
Author: danielwangow daomiao.wang@live.com
Date: 2025-06-17 17:32:20
LastEditors: danielwangow daomiao.wang@live.com
LastEditTime: 2025-09-01 14:28:30
FilePath: /TDA-Homology/utils/dataReader.py
Description: Data IO utilities for anomaly detection experiments.
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical class mapping used across the project
CLASS_NAME_TO_ID: Dict[str, int] = {
    'none': 0,
    'skew': 1,
    'wander': 2,
    'scale': 3,
}
CLASS_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CLASS_NAME_TO_ID.items()}

def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def generate_noise_indicator(signal: np.ndarray, noise_positions) -> np.ndarray:
    """Backward-compatible helper used by notebooks for quick visualization.

    Parameters
    - signal: 1D time series array
    - noise_positions: list of [start, end] or a single tuple

    Returns
    - indicator array with 1 for anomalous indices and 0 otherwise
    """
    noise_indicator = np.zeros_like(signal)
    if noise_positions is None:
        return noise_indicator
    if isinstance(noise_positions, list):
        for start, end in noise_positions:
            start = max(0, int(start))
            end = min(len(signal), int(end))
            if end > start:
                noise_indicator[start:end] = 1
    else:
        start, end = noise_positions
        start = max(0, int(start))
        end = min(len(signal), int(end))
        if end > start:
            noise_indicator[start:end] = 1
    return noise_indicator


def _safe_read_metadata_json(metadata_json_path: str) -> pd.DataFrame:
    """Read the metadata JSON to a DataFrame and normalize key fields.

    Expected keys per row: 'signature' (str), 'signature_locations' (list[[start, end], ...])
    """
    df = pd.read_json(metadata_json_path)
    # Normalize column names if necessary
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()
        if low == 'signature' and col != 'signature':
            rename_map[col] = 'signature'
        if low in ('signature_locations', 'signaturelocation', 'signature_location') and col != 'signature_locations':
            rename_map[col] = 'signature_locations'
    if rename_map:
        df = df.rename(columns=rename_map)
    if 'signature' not in df.columns:
        raise KeyError("metadata JSON must include 'signature' per row")
    # Ensure locations present for non-'none'
    if 'signature_locations' not in df.columns:
        df['signature_locations'] = [[] for _ in range(len(df))]
    return df


def _safe_read_data_csv(data_csv_path: str) -> pd.DataFrame:
    """Read the data CSV where each row is a sample and columns are [id, x0, x1, ...]."""
    df = pd.read_csv(data_csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Data CSV at {data_csv_path} must have >= 2 columns (id + series)")
    return df


def _build_indicator(length: int, spans: Optional[Iterable[Tuple[int, int]]]) -> np.ndarray:
    indicator = np.zeros(length, dtype=np.uint8)
    if not spans:
        return indicator
    for start, end in spans:
        s = max(0, int(start))
        e = min(length, int(end))
        if e > s:
            indicator[s:e] = 1
    return indicator


def iter_samples(
    data_csv_path: str,
    metadata_json_path: str,
    dataset_name: Optional[str] = None,
) -> Iterator[Dict]:
    """Yield per-sample dicts joining data rows with metadata labels.

    Yields dict with keys:
    - dataset: dataset name or derived from parent directory
    - sample_id: value in first column of CSV
    - class_name: one of {'none','skew','wander','scale'}
    - class_id: 0..3 according to CLASS_NAME_TO_ID
    - series: np.ndarray of shape (T,)
    - spans: list of (start,end) anomaly intervals; [] for 'none'
    - labels: np.ndarray of shape (T,) binary 0/1
    """
    data_df = _safe_read_data_csv(data_csv_path)
    meta_df = _safe_read_metadata_json(metadata_json_path)

    if len(data_df) != len(meta_df):
        raise ValueError(
            f"Row count mismatch: data rows={len(data_df)} vs meta rows={len(meta_df)}"
        )

    inferred_dataset_name = dataset_name or os.path.basename(os.path.dirname(data_csv_path))
    for row_idx in range(len(data_df)):
        id_value = data_df.iloc[row_idx, 0]
        series = data_df.iloc[row_idx, 1:].to_numpy(dtype=float)

        class_name = str(meta_df.iloc[row_idx]['signature']).lower().strip()
        if class_name not in CLASS_NAME_TO_ID:
            # Some datasets may use other textual forms; map them conservatively
            if class_name in ('normal', 'clean', 'none'):
                class_name = 'none'
            elif class_name in ('baseline_wander', 'bw', 'wander'):
                class_name = 'wander'
            elif class_name in ('amplitude_scale', 'scale', 'cutoff', 'flip'):
                class_name = 'scale'
            elif class_name in ('skewness','skew'):
                class_name = 'skew'
            else:
                # Fallback: treat unknown as 'none' so downstream filters can exclude
                class_name = 'none'

        spans = meta_df.iloc[row_idx].get('signature_locations', [])
        # Normalize spans to list of tuples
        if isinstance(spans, (list, tuple)) and len(spans) > 0 and isinstance(spans[0], (list, tuple)):
            norm_spans = [(int(s), int(e)) for s, e in spans]
        elif isinstance(spans, (list, tuple)) and len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
            norm_spans = [(int(spans[0]), int(spans[1]))]
        else:
            norm_spans = []

        labels = _build_indicator(length=len(series), spans=norm_spans)

        yield {
            'dataset': inferred_dataset_name,
            'sample_id': id_value,
            'class_name': class_name,
            'class_id': CLASS_NAME_TO_ID[class_name],
            'series': series,
            'spans': norm_spans,
            'labels': labels,
        }


def compute_median_anomaly_length(metadata_json_path: str) -> int:
    """Compute the median length of labeled anomalous spans (ignoring 'none')."""
    meta_df = _safe_read_metadata_json(metadata_json_path)
    lengths: List[int] = []
    for _, r in meta_df.iterrows():
        signature = str(r['signature']).lower().strip()
        if signature == 'none':
            continue
        spans = r.get('signature_locations', [])
        if isinstance(spans, (list, tuple)) and len(spans) > 0 and isinstance(spans[0], (list, tuple)):
            for s, e in spans:
                lengths.append(max(0, int(e) - int(s)))
        elif isinstance(spans, (list, tuple)) and len(spans) == 2:
            s, e = spans
            lengths.append(max(0, int(e) - int(s)))
    if not lengths:
        return 100  # safe default
    return int(np.median(lengths))


def group_counts_by_class(metadata_json_path: str) -> Dict[str, int]:
    """Return the number of samples per class based on metadata."""
    meta_df = _safe_read_metadata_json(metadata_json_path)
    counts = meta_df['signature'].str.lower().value_counts().to_dict()
    # Ensure all classes present
    return {k: int(counts.get(k, 0)) for k in CLASS_NAME_TO_ID.keys()}


def iter_tsb_samples(
    data_pkl_path: str,
    metadata_json_path: str,
    dataset_name: Optional[str] = None,
) -> Iterator[Dict]:
    """Yield per-sample dicts from TSB-UAD-S format (pkl + json).

    Args:
        data_pkl_path: Path to .pkl file containing list of data dictionaries
        metadata_json_path: Path to .json file containing metadata DataFrame
        dataset_name: Optional name for the dataset

    Yields dict with keys:
        - dataset: dataset name or derived from parent directory
        - sample_id: index in the data list
        - class_name: derived from signature_locations (none if empty, else anomaly)
        - class_id: 0..3 according to CLASS_NAME_TO_ID 
        - series: np.ndarray of shape (T,)
        - spans: list of (start,end) anomaly intervals; [] for normal
        - labels: np.ndarray of shape (T,) binary 0/1
    """
    # Load pickle data
    with open(data_pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    
    # Load metadata JSON
    meta_df = pd.read_json(metadata_json_path)
    
    if len(data_list) != len(meta_df):
        raise ValueError(
            f"Data count mismatch: pkl samples={len(data_list)} vs metadata rows={len(meta_df)}"
        )
    
    inferred_dataset_name = dataset_name or os.path.basename(os.path.dirname(data_pkl_path))
    
    for idx in range(len(data_list)):
        # Extract time series data
        data_entry = data_list[idx]
        if isinstance(data_entry, dict) and 'data' in data_entry:
            series = np.array(data_entry['data'], dtype=float)
        elif isinstance(data_entry, (list, np.ndarray)):
            series = np.array(data_entry, dtype=float)
        else:
            raise ValueError(f"Unexpected data format at index {idx}: {type(data_entry)}")
        
        # Extract metadata
        meta_row = meta_df.iloc[idx]
        signature_locations = meta_row.get('signature_locations', [])
        
        # Determine class based on signature locations presence
        # For TSB-UAD-S, keep it simple: samples with signature_locations are anomalous
        if not signature_locations or signature_locations == []:
            class_name = 'none'
            norm_spans = []
        else:
            # All samples with signature_locations are treated as 'skew' (single anomaly class)
            class_name = 'skew'
            
            # Normalize spans to list of tuples
            if isinstance(signature_locations, (list, tuple)) and len(signature_locations) > 0:
                if isinstance(signature_locations[0], (list, tuple)):
                    norm_spans = [(int(s), int(e)) for s, e in signature_locations]
                elif len(signature_locations) == 2 and all(isinstance(x, (int, float)) for x in signature_locations):
                    norm_spans = [(int(signature_locations[0]), int(signature_locations[1]))]
                else:
                    norm_spans = []
            else:
                norm_spans = []
        
        # Generate binary labels
        labels = _build_indicator(length=len(series), spans=norm_spans)
        
        yield {
            'dataset': inferred_dataset_name,
            'sample_id': idx,
            'class_name': class_name,
            'class_id': CLASS_NAME_TO_ID[class_name],
            'series': series,
            'spans': norm_spans,
            'labels': labels,
        }


