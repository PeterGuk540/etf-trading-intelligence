"""CatBoost 3-class classification wrapper (Friend 1 / VVATSO1V params).

Bins the continuous target into 3 classes (down / flat / up) for training,
then converts class probabilities back into a continuous regression-scale
score: (P_up - P_down) * threshold * scale_factor.
"""

import numpy as np
from typing import Optional, Dict, List

from catboost import CatBoostClassifier

from src.models.base_model import BaseModelWrapper


def _bin_target(y: np.ndarray, threshold: float = 0.01):
    """Bin continuous returns into 3 classes: 0=down (<=−1%), 1=flat, 2=up (>1%)."""
    labels = np.ones(len(y), dtype=int)  # default: flat (1)
    labels[y <= -threshold] = 0           # down
    labels[y > threshold] = 2             # up
    return labels


class CatBoostWrapper(BaseModelWrapper):
    """CatBoost 3-class classifier producing a continuous score.

    Hyperparams from Friend 1:
        iterations=1000, learning_rate=0.05, depth=6, loss=MultiClass.

    Parameters
    ----------
    threshold : float
        Boundary for up/down bins (default 0.01 = 1%).
    scale_factor : float
        Multiplier to bring (P_up − P_down) into the same magnitude as
        the regression targets (~±0.05). Configurable for tuning.
    """

    def __init__(self, threshold: float = 0.01, scale_factor: float = 0.05):
        self._threshold = threshold
        self._scale_factor = scale_factor
        self._model: Optional[CatBoostClassifier] = None
        self._feature_names: Optional[List[str]] = None

    # -- properties -----------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "catboost"

    @property
    def model_family(self) -> str:
        return "gbm"

    @property
    def output_type(self) -> str:
        return "classification"

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
        y_binned = _bin_target(y, self._threshold)

        # Ensure all 3 classes are present; if not, duplicate a tiny sample
        unique_classes = set(y_binned)
        if len(unique_classes) < 3:
            # Very rare edge case — just skip training (predict will return 0s)
            print(f"    catboost: only {len(unique_classes)} classes present, "
                  f"skipping training for {sector}")
            self._model = None
            return

        self._model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            verbose=0,
            random_seed=42,
            thread_count=-1,
        )

        eval_set = None
        if val_X is not None and val_y is not None:
            eval_set = (val_X, _bin_target(val_y, self._threshold))

        self._model.fit(
            X_scaled,
            y_binned,
            eval_set=eval_set,
            early_stopping_rounds=100 if eval_set is not None else None,
        )

    def predict(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X_scaled))

        proba = self._model.predict_proba(X_scaled)  # (n, 3) → [P_down, P_flat, P_up]
        p_down = proba[:, 0]
        p_up = proba[:, 2]
        return (p_up - p_down) * self._threshold * self._scale_factor

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or self._feature_names is None:
            return None
        importances = self._model.get_feature_importance()
        return dict(zip(self._feature_names, importances.astype(float)))
