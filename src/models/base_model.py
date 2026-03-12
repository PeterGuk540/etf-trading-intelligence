"""Abstract base class for all 7 ensemble model wrappers."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, List


class BaseModelWrapper(ABC):
    """
    Unified interface that all 7 models implement.

    All wrappers receive flat 2D input (n_samples, 219 features).
    Each wrapper handles its own internal transformations (e.g., sliding window
    for PyTorch models, feature selection for SARIMAX, target binning for CatBoost).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier, e.g. 'lstm', 'lightgbm'."""
        ...

    @property
    @abstractmethod
    def model_family(self) -> str:
        """One of: 'pytorch', 'gbm', 'statsmodels'."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """One of: 'regression', 'classification'."""
        ...

    @abstractmethod
    def train(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sector: str,
        feature_names: List[str],
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_samples, n_features)
            Flat 2D scaled feature matrix.
        y : np.ndarray, shape (n_samples,)
            Continuous regression target (21-day relative return).
        sector : str
            ETF ticker, e.g. 'XLE'.
        feature_names : list[str]
            Column names corresponding to the feature dimension.
        val_X : np.ndarray or None
            Optional validation features for early stopping.
        val_y : np.ndarray or None
            Optional validation targets.
        """
        ...

    @abstractmethod
    def predict(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_samples, n_features)
            Flat 2D scaled feature matrix.
        feature_names : list[str]
            Column names (same order as training).

        Returns
        -------
        np.ndarray
            Regression-scale predictions (continuous, ~+-0.05).
        """
        ...

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance dict, or None if not supported."""
        return None
