# ml_pipeline/models/base_model.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseModel(ABC):
    """
    Lightweight interface for ML models used by the pipeline.
    Concrete models should implement fit(), predict() and optionally predict_quantiles().
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return point prediction (e.g., mean/median).
        """
        ...

    def predict_quantiles(self, X: np.ndarray, quantiles: list) -> dict:
        """
        Optional: return dict {q: array} with quantile predictions.
        Default: raise NotImplementedError to indicate not supported.
        """
        raise NotImplementedError
