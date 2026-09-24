import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def random_kfold(n_samples, n_folds=5, random_seed=42):
   
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return list(kf.split(np.arange(n_samples)))


def temporal_holdout(meta, test_fraction=0.2, date_col='date'):
   
    dates = pd.to_datetime(meta[date_col])
    sorted_order = np.argsort(dates.values)
    n_test = int(round(len(meta) * test_fraction))
    test_idx = sorted_order[-n_test:]
    train_idx = sorted_order[:-n_test]
    return [(train_idx, test_idx)]


def temporal_blocked_kfold(meta, n_folds=5, date_col='date'):
   
    dates = pd.to_datetime(meta[date_col])
    sorted_order = np.argsort(dates.values)

    # Spread the remainder across the first folds so block sizes differ by at most 1
    fold_sizes = np.full(n_folds, len(meta) // n_folds, dtype=int)
    fold_sizes[:len(meta) % n_folds] += 1

    splits, cursor = [], 0
    for size in fold_sizes:
        test_idx = sorted_order[cursor:cursor + size]
        train_idx = np.concatenate([sorted_order[:cursor], sorted_order[cursor + size:]])
        splits.append((train_idx, test_idx))
        cursor += size
    return splits


def leave_one_site_out(meta, site_col='location_name'):
   
    sites = sorted(meta[site_col].unique())
    splits = []
    for site in sites:
        test_mask = (meta[site_col] == site).values
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits