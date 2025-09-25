# tune_randomized.py  (or paste into ml_pipeline/cli.py)
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import scipy.stats as st

def tune_xgb_randomized(X, y, n_iter=60, cv_folds=4, n_jobs=4, random_state=42):
    """
    Randomized search for XGBRegressor. Returns best_params (dict).
    y is expected to be the log10-transformed target (as in your pipeline).
    """
    param_dist = {
        "n_estimators": st.randint(50, 1000),
        "max_depth": st.randint(2, 9),
        "learning_rate": st.loguniform(1e-3, 0.3),
        "subsample": st.uniform(0.5, 0.5),           # [0.5, 1.0)
        "colsample_bytree": st.uniform(0.4, 0.6),    # [0.4, 1.0) but narrower initially
        "min_child_weight": st.randint(1, 20),
        "reg_alpha": st.loguniform(1e-8, 10.0),
        "reg_lambda": st.loguniform(1e-8, 10.0),
        "gamma": st.uniform(0.0, 3.0),
    }

    xgb = XGBRegressor(tree_method="hist", verbosity=0, random_state=random_state)
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    # Note: greater_is_better=False because sklearn expects "bigger is better" by default.
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(xgb, param_distributions=param_dist,
                            n_iter=n_iter, scoring=rmse_scorer,
                            cv=cv, verbose=2, n_jobs=n_jobs, random_state=random_state)

    rs.fit(X, y)
    best = rs.best_params_
    # Convert n_estimators to int if it's a numpy int
    best['n_estimators'] = int(best.get('n_estimators', xgb.get_params()['n_estimators']))
    return best
