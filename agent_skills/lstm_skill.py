from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent_skills.base_skill import BaseSkill, SkillResult
from agent_skills.feature_engineering_skill import FeatureEngineeringSkill
from datasets.walk_forward_splitter import WalkForwardSplitter
from evaluation.metric_registry import metric_registry
from tools.logger import get_logger

log = get_logger(__name__)
_fe = FeatureEngineeringSkill()


class LSTMSkill(BaseSkill):
    name = "lstm"
    description = "PyTorch LSTM regressor with walk-forward validation for price forecasting."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("pip install torch") from exc

        data: pd.DataFrame = params["data"]
        train_pct: float = params.get("train_pct", 0.80)
        n_folds: int = params.get("n_folds", 5)
        metric_name: str = params.get("metric_name", "mape")
        target_col: str = params.get("target_col", "close")
        fe_params: dict = params.get("fe_params", {})

        seq_len: int = params.get("sequence_length", 30)
        hidden_size: int = params.get("hidden_size", 64)
        n_layers: int = params.get("n_layers", 2)
        dropout: float = params.get("dropout", 0.2)
        epochs: int = params.get("epochs", 30)
        batch_size: int = params.get("batch_size", 32)
        lr: float = params.get("learning_rate", 0.001)

        enriched = _fe.transform(data, fe_params)
        feature_cols = [c for c in enriched.columns if c != target_col]
        n_features = len(feature_cols)

        splitter = WalkForwardSplitter(train_pct=train_pct, n_folds=n_folds)
        folds = splitter.split(enriched)

        fold_scores: list[float] = []
        all_preds: list[np.ndarray] = []
        last_model = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for fold in folds:
            if len(fold.train) <= seq_len or len(fold.val) <= seq_len:
                log.warning("fold_too_small_for_lstm", fold=fold.fold_index, seq_len=seq_len)
                continue

            X_train_seq, y_train_seq = _make_sequences(
                fold.train[feature_cols].values,
                fold.train[target_col].values,
                seq_len,
            )
            X_val_seq, y_val_seq = _make_sequences(
                fold.val[feature_cols].values,
                fold.val[target_col].values,
                seq_len,
            )

            model = _LSTMModel(n_features, hidden_size, n_layers, dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train_ds = TensorDataset(
                torch.FloatTensor(X_train_seq).to(device),
                torch.FloatTensor(y_train_seq).to(device),
            )
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

            model.train()
            for _ in range(epochs):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch).squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.FloatTensor(X_val_seq).to(device)
                ).squeeze().cpu().numpy()

            score = metric_registry.evaluate(metric_name, y_val_seq, preds)
            fold_scores.append(score)
            all_preds.append(preds)
            last_model = model

        if not fold_scores:
            raise RuntimeError("All folds were too small for the configured sequence_length.")

        all_predictions = np.concatenate(all_preds)
        val_score = float(np.mean(fold_scores))

        last_fold = folds[-1]
        if len(last_fold.train) > seq_len:
            X_tr, y_tr = _make_sequences(
                last_fold.train[feature_cols].values,
                last_fold.train[target_col].values,
                seq_len,
            )
            last_model.eval()
            with torch.no_grad():
                tr_preds = last_model(
                    torch.FloatTensor(X_tr).to(device)
                ).squeeze().cpu().numpy()
            train_score = metric_registry.evaluate(metric_name, y_tr, tr_preds)
        else:
            train_score = val_score

        log.info("lstm_trained", val_score=round(val_score, 4), folds=len(fold_scores))

        return SkillResult(
            skill_name=self.name,
            predictions=all_predictions,
            fold_scores=fold_scores,
            train_score=train_score,
            val_score=val_score,
            model=last_model,
            params_used={
                "hidden_size": hidden_size, "n_layers": n_layers,
                "dropout": dropout, "sequence_length": seq_len,
                "epochs": epochs, "learning_rate": lr,
            },
            metadata={"feature_cols": feature_cols, "n_folds": n_folds},
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "data": "pd.DataFrame (OHLCV)",
                "train_pct": "float (default 0.80)",
                "n_folds": "int (default 5)",
                "sequence_length": "int (default 30)",
                "hidden_size": "int (default 64)",
                "n_layers": "int (default 2)",
                "dropout": "float (default 0.2)",
                "epochs": "int (default 30)",
                "learning_rate": "float (default 0.001)",
            }
        }


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


class _LSTMModel:
    def __new__(cls, *args, **kwargs):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("pip install torch") from exc
        return _build_lstm_model(*args, **kwargs)


def _build_lstm_model(input_size: int, hidden_size: int, n_layers: int, dropout: float):
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return Model()
