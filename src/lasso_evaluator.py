import time
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def run_lasso_with_splits(target_name, meta, otu, splits,
                           strategy_name="custom",
                           input_name="unknown",
                           log_target=False,
                           random_seed=42,
                           verbose=True):
    
    # ---- Prepare X and y from the already-filtered tables ----
    X = otu.values
    y = meta[target_name].values
    if log_target:
        y = np.log1p(y)

    n_folds = len(splits)
    target_label = target_name + (" (log1p)" if log_target else "")

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  Lasso · input={input_name} · target={target_label} · strategy={strategy_name}")
        print(f"{'=' * 64}")
        print(f"  Samples: {len(y)}    Features: {X.shape[1]}    Folds: {n_folds}")

    # ---- Per-fold containers ----
    fold_r2, fold_mae, fold_rmse = [], [], []
    fold_alpha, fold_n_selected, fold_time = [], [], []

    # ---- Outer loop over the supplied splits ----
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fresh pipeline per fold — no leakage
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso',  LassoCV(cv=5,
                               random_state=random_seed,
                               max_iter=10_000,
                               n_jobs=-1)),
        ])

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_hat = pipe.predict(X_te)
        r2    = r2_score(y_te, y_hat)
        mae   = mean_absolute_error(y_te, y_hat)
        rmse  = np.sqrt(mean_squared_error(y_te, y_hat))
        alpha = pipe.named_steps['lasso'].alpha_
        n_sel = int((pipe.named_steps['lasso'].coef_ != 0).sum())

        fold_r2.append(r2);   fold_mae.append(mae);   fold_rmse.append(rmse)
        fold_alpha.append(alpha); fold_n_selected.append(n_sel); fold_time.append(elapsed)

        if verbose:
            print(f"    Fold {fold_idx:2d}: R²={r2:7.4f}  MAE={mae:6.3f}  "
                  f"α={alpha:.4f}  feat={n_sel}  time={elapsed:.1f}s")

    # ---- Aggregate ----
    results = {
        'strategy':        strategy_name,
        'input':           input_name,
        'target':          target_label,
        'log_target':      log_target,
        'n_folds':         n_folds,
        'n_samples':       int(len(y)),
        'n_features':      int(X.shape[1]),
        'r2_mean':         float(np.mean(fold_r2)),
        'r2_std':          float(np.std(fold_r2)),
        'mae_mean':        float(np.mean(fold_mae)),
        'mae_std':         float(np.std(fold_mae)),
        'rmse_mean':       float(np.mean(fold_rmse)),
        'rmse_std':        float(np.std(fold_rmse)),
        'alpha_mean':      float(np.mean(fold_alpha)),
        'n_selected_mean': float(np.mean(fold_n_selected)),
        'total_time':      float(np.sum(fold_time)),
    }

    if verbose:
        print(f"\n  Summary:")
        print(f"    R²:   {results['r2_mean']:.4f}  ±  {results['r2_std']:.4f}")
        print(f"    MAE:  {results['mae_mean']:.3f}  ±  {results['mae_std']:.3f}")
        print(f"    Mean α:              {results['alpha_mean']:.4f}")
        print(f"    Mean features used:  {results['n_selected_mean']:.0f} / {X.shape[1]}")
        print(f"    Total CV time:       {results['total_time']:.1f}s")

    return results