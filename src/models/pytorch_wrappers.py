"""Thin wrappers around existing PyTorch nn.Module classes to conform to BaseModelWrapper.

Handles flat -> 3D sequence conversion internally via sliding window (seq_len=20).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List

from src.models.base_model import BaseModelWrapper


def _create_sequences(X: np.ndarray, y: Optional[np.ndarray], seq_len: int):
    """Convert flat (n, f) -> (n-seq_len, seq_len, f) sliding window sequences."""
    X_seq = []
    y_seq = [] if y is not None else None
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        if y is not None:
            y_seq.append(y[i])
    X_seq = np.array(X_seq) if X_seq else np.empty((0, seq_len, X.shape[1]))
    if y is not None:
        y_seq = np.array(y_seq) if y_seq else np.empty((0,))
    return X_seq, y_seq


class PyTorchModelWrapper(BaseModelWrapper):
    """Wraps any nn.Module that expects 3D (batch, seq_len, features) input.

    Parameters
    ----------
    module_class : type
        An nn.Module subclass (SimpleLSTM, SimpleTFT, SimpleNBeats, SimpleLSTMGARCH).
    name : str
        Short model identifier, e.g. 'lstm'.
    seq_len : int
        Sliding window length (default 20).
    epochs : int
        Training epochs (default 50).
    lr : float
        Learning rate (default 0.001).
    """

    def __init__(
        self,
        module_class: type,
        name: str,
        seq_len: int = 20,
        epochs: int = 50,
        lr: float = 0.001,
    ):
        self._module_class = module_class
        self._name = name
        self._seq_len = seq_len
        self._epochs = epochs
        self._lr = lr
        self._model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None

    # -- BaseModelWrapper properties ------------------------------------------

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_family(self) -> str:
        return "pytorch"

    @property
    def output_type(self) -> str:
        return "regression"

    # -- BaseModelWrapper methods ---------------------------------------------

    def train(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sector: str,
        feature_names: List[str],
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> None:
        self._input_dim = X_scaled.shape[1]

        # Build sequences
        X_seq, y_seq = _create_sequences(X_scaled, y, self._seq_len)
        if len(X_seq) < 2:
            raise ValueError(
                f"{self._name}: not enough samples after sequencing "
                f"({len(X_scaled)} rows, seq_len={self._seq_len})"
            )

        # Instantiate model
        if self._name == "nbeats":
            self._model = self._module_class(self._input_dim, seq_len=self._seq_len)
        else:
            self._model = self._module_class(self._input_dim)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_seq)
        y_t = torch.FloatTensor(y_seq)

        self._model.train()
        for epoch in range(self._epochs):
            optimizer.zero_grad()
            output = self._model(X_t)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            loss = criterion(output, y_t)
            loss.backward()
            optimizer.step()

    def predict(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self._name}: model not trained yet")

        X_seq, _ = _create_sequences(X_scaled, None, self._seq_len)
        if len(X_seq) == 0:
            # Fallback: use all available rows as a single sequence (pad if needed)
            if len(X_scaled) >= 1:
                padded = np.zeros((self._seq_len, X_scaled.shape[1]))
                padded[-len(X_scaled) :] = X_scaled[: self._seq_len]
                X_seq = padded[np.newaxis, ...]
            else:
                return np.array([0.0])

        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.FloatTensor(X_seq))
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            return preds.numpy()
