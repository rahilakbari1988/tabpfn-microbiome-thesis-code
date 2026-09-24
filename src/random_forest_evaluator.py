
import time
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 12-combination grid — same size as the XGBoost grid for fair comparison.
# n_estimators × max_depth × min_samples_leaf = 2 × 3 × 2 = 12.
RF_PARAM_GRID = {
    'n_estimators':     [200, 500],
    'max_depth':        [None, 10, 20],
    'min_samples_leaf': [1, 5],
}


def _make_rf(random_seed=42):
    return RandomForestRegressor(
        random_state=random_seed,
        n_jobs=-1,
        bootstrap=True,
    )


def run_random_forest_with_splits(
    target_name,
    meta,
    otu,
    splits,
    strategy_name,
    input_name,
    log_target,
    random_seed=42,
    verbose=True,
):
    
    target_label = f"{target_name}{' (log1p)' if log_target else ''}"

    X_all = otu.values
    y_raw = meta[target_name].values
    y_all = np.log1p(y_raw) if log_target else y_raw

    n_samples, n_features = X_all.shape

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Target: {target_label}    Input features: {n_features}")
        print(f"{'=' * 60}")
        print(f"  Samples: {n_samples}    Folds: {len(splits)}")
        print(f"  Strategy: {strategy_name}    Input: {input_name}")

    # Per-fold accumulators
    fold_r2, fold_mae, fold_rmse, fold_time = [], [], [], []
    fold_best_n_est, fold_best_max_depth, fold_best_min_samples_leaf = [], [], []

    t_total_start = time.time()

    for k, (tr_idx, te_idx) in enumerate(splits, start=1):
        t0 = time.time()

        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # Inner 5-fold CV over the param grid — structural equivalent of
        # LassoCV's alpha tuning and XGBoost's GridSearchCV.
        gs = GridSearchCV(
            estimator=_make_rf(random_seed),
            param_grid=RF_PARAM_GRID,
            cv=5,
            scoring='r2',
            n_jobs=1,        # n_jobs is already on the RF estimator
            refit=True,
        )
        gs.fit(X_tr, y_tr)

        best_params = gs.best_params_
        y_pred = gs.predict(X_te)

        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        dt   = time.time() - t0

        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_time.append(dt)
        fold_best_n_est.append(best_params['n_estimators'])
        fold_best_max_depth.append(best_params['max_depth'])
        fold_best_min_samples_leaf.append(best_params['min_samples_leaf'])

        if verbose:
            print(f"  fold {k:2d}/{len(splits)}  "
                  f"R²={r2:+.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
                  f"time={dt:.1f}s  best={best_params}")

    total_time = time.time() - t_total_start

    # Mode (most-frequent value) of the per-fold best params.
    # We use Counter rather than pd.Series.mode because max_depth=None
    # can confuse pandas in some versions.
    def _mode(seq):
        return Counter(seq).most_common(1)[0][0]

    results = {
        # Identifiers
        'target':     target_name,
        'input':      input_name,
        'strategy':   strategy_name,
        'log_target': log_target,
        # Sizes
        'n_samples':  int(n_samples),
        'n_features': int(n_features),
        'n_folds':    len(splits),
        # Summary metrics (mean ± std across folds)
        'r2_mean':    float(np.mean(fold_r2)),
        'r2_std':     float(np.std(fold_r2)),
        'mae_mean':   float(np.mean(fold_mae)),
        'mae_std':    float(np.std(fold_mae)),
        'rmse_mean':  float(np.mean(fold_rmse)),
        'rmse_std':   float(np.std(fold_rmse)),
        # Model-specific: mode of best params across folds
        'best_n_est_mode':            _mode(fold_best_n_est),
        'best_max_depth_mode':        _mode(fold_best_max_depth),
        'best_min_samples_leaf_mode': _mode(fold_best_min_samples_leaf),
        # Per-fold details (lists, kept in memory only — not in the CSV)
        'fold_r2':                    fold_r2,
        'fold_mae':                   fold_mae,
        'fold_rmse':                  fold_rmse,
        'fold_time':                  fold_time,
        'fold_best_n_est':            fold_best_n_est,
        'fold_best_max_depth':        fold_best_max_depth,
        'fold_best_min_samples_leaf': fold_best_min_samples_leaf,
        # Timing
        'total_time': total_time,
    }

    if verbose:
        print(f"\n  Summary · {target_label} · {input_name} · {strategy_name}")
        print(f"    R²:    {results['r2_mean']:+.4f}  ±  {results['r2_std']:.4f}")
        print(f"    MAE:   {results['mae_mean']:.3f}  ±  {results['mae_std']:.3f}")
        print(f"    RMSE:  {results['rmse_mean']:.3f}  ±  {results['rmse_std']:.3f}")
        print(f"    Most common best params: "
              f"n_est={results['best_n_est_mode']}, "
              f"max_depth={results['best_max_depth_mode']}, "
              f"min_samples_leaf={results['best_min_samples_leaf_mode']}")
        print(f"    Total CV time: {results['total_time']:.1f}s")

    return results