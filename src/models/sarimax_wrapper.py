"""SARIMAX wrapper (Friend 2 / zhoyi11 params).

Internally selects the top 15 features per sector by absolute Pearson
correlation with the target, applies StandardScaler, then fits SARIMAX
with order=(2,0,1) and no seasonal component.

Graceful failure: if SARIMAX fails to converge for a sector, returns zeros
so the ensemble proceeds with the remaining 6 models.
"""

import numpy as np
from typing import Optional, Dict, List
from sklearn.preprocessing import StandardScaler
import warnings

from src.models.base_model import BaseModelWrapper


class SARIMAXWrapper(BaseModelWrapper):
    """SARIMAX time-series wrapper.

    Parameters from Friend 2:
        order=(2,0,1), seasonal_order=(0,0,0,0),
        enforce_stationarity=False, enforce_invertibility=False,
        maxiter=200, top 15 features by |correlation|.
    """

    def __init__(self, n_top_features: int = 15):
        self._n_top = n_top_features
        self._selected_features: Optional[List[str]] = None
        self._selected_indices: Optional[List[int]] = None
        self._scaler: Optional[StandardScaler] = None
        self._results = None  # statsmodels SARIMAXResults
        self._trained = False

    # -- properties -----------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "sarimax"

    @property
    def model_family(self) -> str:
        return "statsmodels"

    @property
    def output_type(self) -> str:
        return "regression"

    # -- helpers --------------------------------------------------------------

    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[int]:
        """Select top-N features by absolute Pearson correlation with target."""
        correlations = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.std(col) < 1e-12:
                correlations.append(0.0)
            else:
                correlations.append(abs(np.corrcoef(col, y)[0, 1]))

        correlations = np.array(correlations)
        # Replace NaN correlations with 0
        correlations = np.nan_to_num(correlations, nan=0.0)

        top_indices = np.argsort(correlations)[::-1][: self._n_top]
        self._selected_features = [feature_names[i] for i in top_indices]
        self._selected_indices = top_indices.tolist()
        return self._selected_indices

    # -- train / predict ------------------------------------------------------

    def train(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sector: str,
        feature_names: List[str],
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> None:
        # Lazy import — statsmodels is heavy
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._trained = False

        # 1. Feature selection
        indices = self._select_features(X_scaled, y, feature_names)
        X_subset = X_scaled[:, indices]

        # 2. Rescale the subset (SARIMAX benefits from its own scaling)
        self._scaler = StandardScaler()
        X_sub_scaled = self._scaler.fit_transform(X_subset)

        # 3. Fit SARIMAX
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    endog=y,
                    exog=X_sub_scaled,
                    order=(2, 0, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._results = model.fit(maxiter=200, disp=False)
                self._trained = True
        except Exception as e:
            print(f"    sarimax: failed to converge for {sector} — {e}")
            self._results = None

    def predict(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        if not self._trained or self._results is None:
            return np.zeros(len(X_scaled))

        X_subset = X_scaled[:, self._selected_indices]
        X_sub_scaled = self._scaler.transform(X_subset)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = self._results.forecast(
                    steps=len(X_scaled),
                    exog=X_sub_scaled,
                )
                preds = np.asarray(preds, dtype=float)
                # Clamp explosive forecasts (unstable AR roots)
                preds = np.clip(preds, -1.0, 1.0)
                return preds
        except Exception:
            return np.zeros(len(X_scaled))

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._trained or self._results is None or self._selected_features is None:
            return None
        try:
            params = self._results.params
            # Exog coefficients start after the ARMA params (const + AR + MA terms)
            # order=(2,0,1) → const(1) + AR(2) + MA(1) = 4 params before exog
            n_pre = 4
            exog_params = params[n_pre : n_pre + len(self._selected_features)]
            return dict(zip(self._selected_features, np.abs(exog_params).tolist()))
        except Exception:
            return None
