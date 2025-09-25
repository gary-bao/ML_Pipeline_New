# ml_pipeline/models/xgb_model.py
from typing import Optional, List, Dict
import numpy as np
import joblib
import logging
from .base_model import BaseModel
import xgboost as xgb
from xgboost import XGBRegressor

class XGBModel(BaseModel):
    """
    Wrapper for XGBoost regressor(s).
    - If quantiles is None -> trains a single XGBRegressor (mean/point predictions).
    - If quantiles provided (list of floats), trains a separate quantile model per alpha
      using objective='reg:quantileerror' and quantile_alpha.
    """

    def __init__(self, params: Optional[dict] = None, n_estimators: int = 100):
        self.params = params or {}
        self.n_estimators = n_estimators
        self._model = None                # for point model
        self._quantile_models = {}        # for quantile models keyed by alpha

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: Optional[List[float]] = None) -> None:
        if quantiles:
            logging.info(f"Training {len(quantiles)} quantile XGBoost models for alphas={quantiles}")
            self._quantile_models = {}
            for q in quantiles:
                # use low-level xgb.train with QuantileDMatrix OR XGBRegressor with objective param
                params = dict(self.params)
                params.update({"objective": "reg:quantileerror", "quantile_alpha": float(q)})
                model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=self.n_estimators)
                self._quantile_models[q] = model
            # no `_model` in quantile-mode
            self._model = None
        else:
            # point model
            logging.info("Training XGBRegressor (point predictions)")
            self._model = XGBRegressor(n_estimators=self.n_estimators, **(self.params or {}))
            self._model.fit(X, y)
            self._quantile_models = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Point model not trained. Train with quantiles=None or call predict_quantiles.")
        return self._model.predict(X)

    def predict_quantiles(self, X: np.ndarray, quantiles: List[float]) -> Dict[float, np.ndarray]:
        if not self._quantile_models:
            raise RuntimeError("Quantile models not trained. Call fit(..., quantiles=...) first.")
        result = {}
        dmat = xgb.DMatrix(X)
        for q in quantiles:
            model = self._quantile_models.get(q)
            if model is None:
                raise KeyError(f"No model for quantile {q}")
            preds = model.predict(dmat)
            result[q] = preds
        return result

    def save(self, path: str) -> None:
        # save both point and quantile models
        joblib.dump({"point": self._model, "quantiles": self._quantile_models}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self._model = obj.get("point", None)
        self._quantile_models = obj.get("quantiles", {})
