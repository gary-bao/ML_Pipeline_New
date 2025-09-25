# tune_optuna.py (or paste into ml_pipeline/cli.py)
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def optuna_objective(trial, X, y, cv_splits=3, random_state=42):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 9),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "gamma": trial.suggest_uniform("gamma", 0.0, 5.0),
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": random_state
    }

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        model = XGBRegressor(**params)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        pred = model.predict(Xval)
        rmse = np.sqrt(mean_squared_error(yval, pred))
        rmses.append(rmse)
    return float(np.mean(rmses))


def tune_xgb_optuna(X, y, n_trials=50, cv_splits=3, n_jobs=1, random_state=42):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    func = lambda trial: optuna_objective(trial, X, y, cv_splits=cv_splits, random_state=random_state)
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params
