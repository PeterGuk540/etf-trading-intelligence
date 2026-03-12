"""LightGBM regression wrapper (Friend 1 / VVATSO1V params).

Uses flat 2D input directly — tree models treat each row independently.
Trains with 5-fold TimeSeriesSplit CV to find optimal n_estimators via early
stopping (patience=200), then retrains on full training data with median
best_iteration.
"""

import numpy as np
from typing import Optional, Dict, List

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.models.base_model import BaseModelWrapper


class LightGBMWrapper(BaseModelWrapper):
    """LightGBM regression model.

    Hyperparams from Friend 1 (VVATSO1V):
        learning_rate=0.03, num_leaves=63, feature_fraction=0.8,
        bagging_fraction=0.8, lambda_l2=1.0, n_estimators=5000 (max),
        early_stopping=200, 5-fold TimeSeriesSplit CV.
    """

    def __init__(self):
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_names: Optional[List[str]] = None
        self._best_iteration: int = 500  # fallback

    # -- properties -----------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "lightgbm"

    @property
    def model_family(self) -> str:
        return "gbm"

    @property
    def output_type(self) -> str:
        return "regression"

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
        self._feature_names = list(feature_names)

        params = dict(
            objective="regression",
            learning_rate=0.03,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            n_estimators=5000,
            verbosity=-1,
        )

        # --- 5-fold TimeSeriesSplit CV to find best n_estimators ---
        tscv = TimeSeriesSplit(n_splits=5)
        best_iterations: List[int] = []

        for train_idx, val_idx in tscv.split(X_scaled):
            fold_X_train, fold_y_train = X_scaled[train_idx], y[train_idx]
            fold_X_val, fold_y_val = X_scaled[val_idx], y[val_idx]

            fold_model = lgb.LGBMRegressor(**params)
            fold_model.fit(
                fold_X_train,
                fold_y_train,
                eval_set=[(fold_X_val, fold_y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iterations.append(fold_model.best_iteration_)

        median_iters = int(np.median(best_iterations))
        self._best_iteration = max(median_iters, 10)

        # --- Retrain on full training data with median best_iteration ---
        params["n_estimators"] = self._best_iteration
        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X_scaled, y)

    def predict(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("lightgbm: model not trained yet")
        return self._model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or self._feature_names is None:
            return None
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.astype(float)))
