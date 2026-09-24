import time
import numpy as np
import pandas as pd

from importlib.metadata import version as _pkg_version
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabpfn import TabPFNRegressor


# Module-level constants
TABPFN_VERSION = _pkg_version("tabpfn")


def _make_tabpfn(random_seed, model_path="auto"):
    return TabPFNRegressor(
        model_path=model_path,
        device='cuda',
        random_state=random_seed,
        ignore_pretraining_limits=True,
    )


def run_tabpfn_with_splits(
    target_name,
    meta,
    otu,
    splits,
    strategy_name,
    input_name,
    log_target,
    random_seed=42,
    model_path="auto",
    verbose=True,
):

    # ---- Prepare X and y ----
    X = otu.values
    y = meta[target_name].values
    if log_target:
        y = np.log1p(y)

    n_folds       = len(splits)
    target_label  = target_name + (" (log1p)" if log_target else "")

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  TabPFN · input={input_name} · target={target_label} · strategy={strategy_name}")
        print(f"{'=' * 64}")
        print(f"  Samples: {len(y)}    Features: {X.shape[1]}    Folds: {n_folds}")
        print(f"  No inner-CV (pretrained model)  ·  TabPFN v{TABPFN_VERSION}")

    # ---- Per-fold containers ----
    fold_r2, fold_mae, fold_rmse, fold_time = [], [], [], []

    # ---- Outer loop ----
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fresh model per fold — no state carry-over between folds.
        model = _make_tabpfn(random_seed, model_path)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        elapsed = time.time() - t0

        r2    = r2_score(y_te, y_hat)
        mae   = mean_absolute_error(y_te, y_hat)
        rmse  = np.sqrt(mean_squared_error(y_te, y_hat))

        fold_r2.append(r2);     fold_mae.append(mae); fold_rmse.append(rmse)
        fold_time.append(elapsed)

        if verbose:
            print(f"    Fold {fold_idx:2d}: R²={r2:+.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
                  f"time={elapsed:.1f}s")

    # ---- Aggregate (same schema as run_xgboost_with_splits, minus best_*) ----
    results = {
        'target':           target_name,
        'input':            input_name,
        'strategy':         strategy_name,
        'log_target':       log_target,
        'n_samples':        int(len(y)),
        'n_features':       int(X.shape[1]),
        'n_folds':          n_folds,
        'r2_mean':          float(np.mean(fold_r2)),
        'r2_std':           float(np.std(fold_r2)),
        'mae_mean':         float(np.mean(fold_mae)),
        'mae_std':          float(np.std(fold_mae)),
        'rmse_mean':        float(np.mean(fold_rmse)),
        'rmse_std':         float(np.std(fold_rmse)),
        'total_time':       float(np.sum(fold_time)),
        # TabPFN-specific (replaces best_*_mode fields)
        'notes':            'no_tuning_pretrained',
        'tabpfn_version':   TABPFN_VERSION,
        'model_path':       model_path,
        # per-fold details
        'fold_r2':          fold_r2,
        'fold_mae':         fold_mae,
        'fold_rmse':        fold_rmse,
        'fold_time':        fold_time,
    }

    if verbose:
        print(f"\n  Summary · {target_label} · {input_name} · {strategy_name}")
        print(f"    R²:    {results['r2_mean']:+.4f}  ±  {results['r2_std']:.4f}")
        print(f"    MAE:   {results['mae_mean']:.3f}  ±  {results['mae_std']:.3f}")
        print(f"    RMSE:  {results['rmse_mean']:.3f}  ±  {results['rmse_std']:.3f}")
        print(f"    TabPFN version:          {results['tabpfn_version']}")
        print(f"    Total CV time:           {results['total_time']:.1f}s")

    return results