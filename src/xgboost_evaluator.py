import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# Inner-CV hyperparameter grid
XGB_PARAM_GRID = {
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators':  [200, 500],
}


def _make_xgb_base(random_seed):
    return XGBRegressor(
        objective='reg:squarederror',
        device='cuda',
        tree_method='hist',
        random_state=random_seed,
        n_jobs=1,
        verbosity=0,
    )


def run_xgboost_with_splits(
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
   
    # ---- Prepare X and y ----
    X = otu.values
    y = meta[target_name].values
    if log_target:
        y = np.log1p(y)

    n_folds       = len(splits)
    target_label  = target_name + (" (log1p)" if log_target else "")

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  XGBoost · input={input_name} · target={target_label} · strategy={strategy_name}")
        print(f"{'=' * 64}")
        print(f"  Samples: {len(y)}    Features: {X.shape[1]}    Folds: {n_folds}")
        print(f"  Inner-CV grid: {sum(1 for _ in _iter_grid(XGB_PARAM_GRID))} combos × 5-fold = "
              f"{sum(1 for _ in _iter_grid(XGB_PARAM_GRID)) * 5} fits per outer fold")

    # ---- Per-fold containers ----
    fold_r2, fold_mae, fold_rmse, fold_time = [], [], [], []
    fold_best_max_depth, fold_best_lr, fold_best_n_est = [], [], []

    # ---- Outer loop ----
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fresh model + inner CV per fold — no leakage
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        grid = GridSearchCV(
            estimator   = _make_xgb_base(random_seed),
            param_grid  = XGB_PARAM_GRID,
            cv          = inner_cv,
            scoring     = 'r2',
            n_jobs      = 1,        # GPU is the bottleneck — see _make_xgb_base
            refit       = True,
            verbose     = 0,
        )

        t0 = time.time()
        grid.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_hat = grid.predict(X_te)
        r2    = r2_score(y_te, y_hat)
        mae   = mean_absolute_error(y_te, y_hat)
        rmse  = np.sqrt(mean_squared_error(y_te, y_hat))

        best = grid.best_params_

        fold_r2.append(r2);     fold_mae.append(mae); fold_rmse.append(rmse)
        fold_time.append(elapsed)
        fold_best_max_depth.append(best['max_depth'])
        fold_best_lr.append(best['learning_rate'])
        fold_best_n_est.append(best['n_estimators'])

        if verbose:
            print(f"    Fold {fold_idx:2d}: R²={r2:+.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
                  f"best=(d={best['max_depth']}, lr={best['learning_rate']}, "
                  f"n={best['n_estimators']})  time={elapsed:.1f}s")

    # ---- Aggregate (same schema as run_lasso_with_splits) ----
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
        # XGBoost-specific (mode = most-frequently-chosen value across folds)
        'best_max_depth_mode': int(pd.Series(fold_best_max_depth).mode().iloc[0]),
        'best_lr_mode':        float(pd.Series(fold_best_lr).mode().iloc[0]),
        'best_n_est_mode':     int(pd.Series(fold_best_n_est).mode().iloc[0]),
        # per-fold details
        'fold_r2':          fold_r2,
        'fold_mae':         fold_mae,
        'fold_rmse':        fold_rmse,
        'fold_best_max_depth': fold_best_max_depth,
        'fold_best_lr':        fold_best_lr,
        'fold_best_n_est':     fold_best_n_est,
        'fold_time':        fold_time,
    }

    if verbose:
        print(f"\n  Summary · {target_label} · {input_name} · {strategy_name}")
        print(f"    R²:    {results['r2_mean']:+.4f}  ±  {results['r2_std']:.4f}")
        print(f"    MAE:   {results['mae_mean']:.3f}  ±  {results['mae_std']:.3f}")
        print(f"    RMSE:  {results['rmse_mean']:.3f}  ±  {results['rmse_std']:.3f}")
        print(f"    Most common best params: "
              f"max_depth={results['best_max_depth_mode']}, "
              f"lr={results['best_lr_mode']}, "
              f"n_est={results['best_n_est_mode']}")
        print(f"    Total CV time:           {results['total_time']:.1f}s")

    return results


# Internal helpers
def _iter_grid(grid):
    """Yield all combinations of a sklearn-style param grid (for logging)."""
    from itertools import product
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))