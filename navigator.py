#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal inference wrapper for the saved weighted-CE navigator MLP.

Expected saved files:
- final_scaler.joblib
- final_label_encoder.joblib
- final_weighted_ce_mlp.pt

Input:
    feature vector with this exact order:
    [sim, rmse, x, y, z, qw, qx, qy, qz]

Output:
    predicted class label
    optional class probabilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from joblib import load


FEATURE_COLUMNS = [
    "sim",
    "rmse",
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
]


class WeightedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(128, 64), dropout=0.10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NavigatorMLPInference:
    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        label_encoder_path: Union[str, Path],
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.scaler = load(self.scaler_path)
        self.label_encoder = load(self.label_encoder_path)

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.feature_columns = checkpoint.get("feature_columns", FEATURE_COLUMNS)
        self.class_names = checkpoint["class_names"]
        self.input_dim = int(checkpoint["input_dim"])
        self.num_classes = int(checkpoint["num_classes"])
        self.hidden_dims = tuple(checkpoint["hidden_dims"])
        self.dropout = float(checkpoint["dropout"])

        self.model = WeightedMLP(
            in_dim=self.input_dim,
            out_dim=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _vector_from_input(self, features: Union[Sequence[float], Dict[str, float]]) -> np.ndarray:
        if isinstance(features, dict):
            missing = [k for k in self.feature_columns if k not in features]
            if missing:
                raise ValueError(f"Missing feature(s): {missing}")
            vec = np.array([features[k] for k in self.feature_columns], dtype=np.float64)
        else:
            vec = np.asarray(features, dtype=np.float64).reshape(-1)
            if vec.shape[0] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} features in this order "
                    f"{self.feature_columns}, but got shape {vec.shape}"
                )

        if not np.isfinite(vec).all():
            raise ValueError(f"Feature vector contains NaN or Inf values: {vec}")

        return vec

    def predict_proba(self, features: Union[Sequence[float], Dict[str, float]]) -> Dict[str, float]:
        vec = self._vector_from_input(features).reshape(1, -1)
        vec_sc = self.scaler.transform(vec)

        x = torch.tensor(vec_sc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {cls: float(p) for cls, p in zip(self.class_names, probs)}

    def predict(self, features: Union[Sequence[float], Dict[str, float]]) -> str:
        proba = self.predict_proba(features)
        return max(proba, key=proba.get)

    def predict_full(self, features: Union[Sequence[float], Dict[str, float]]) -> Dict[str, object]:
        vec = self._vector_from_input(features).reshape(1, -1)
        vec_sc = self.scaler.transform(vec)

        x = torch.tensor(vec_sc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        return {
            "predicted_label": str(self.class_names[pred_idx]),
            "predicted_index": pred_idx,
            "probabilities": {cls: float(p) for cls, p in zip(self.class_names, probs)},
            "feature_order": list(self.feature_columns),
            "input_vector": vec.reshape(-1).tolist(),
        }


if __name__ == "__main__":
    # Example paths
    model_path = "notebook_outputs/real_nav_lopo_weighted_ce/final_weighted_ce_mlp.pt"
    scaler_path = "notebook_outputs/real_nav_lopo_weighted_ce/final_scaler.joblib"
    label_encoder_path = "notebook_outputs/real_nav_lopo_weighted_ce/final_label_encoder.joblib"

    navigator = NavigatorMLPInference(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
    )

    # Example 1: input as a list in the exact expected order
    feature_vector = [
        0.82,   # sim
        0.95,   # rmse
        0.12,   # x
        0.03,   # y
        -0.48,  # z
        0.999,  # qw
        0.010,  # qx
        0.005,  # qy
        -0.020, # qz
    ]

    result = navigator.predict_full(feature_vector)
    print("Example 1")
    print("Predicted label:", result["predicted_label"])
    print("Probabilities:")
    for k, v in result["probabilities"].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "-" * 60 + "\n")

    # Example 2: input as a dictionary
    feature_dict = {
        "sim": 0.76,
        "rmse": 1.10,
        "x": -0.05,
        "y": 0.02,
        "z": -0.20,
        "qw": 0.998,
        "qx": 0.015,
        "qy": -0.004,
        "qz": 0.055,
    }

    pred_label = navigator.predict(feature_dict)
    pred_proba = navigator.predict_proba(feature_dict)

    print("Example 2")
    print("Predicted label:", pred_label)
    print("Probabilities:")
    for k, v in pred_proba.items():
        print(f"  {k}: {v:.4f}")