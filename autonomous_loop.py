#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
autonomous_loop.py

Hybrid autonomous decision loop for the real navigator.

Main responsibilities:
1. Compute the 9D feature vector from:
   - current RGB-D pair
   - key RGB-D pair
2. Run the pretrained weighted MLP navigator
3. Apply an oscillation / loop-buffer heuristic
4. Return a rich decision record:
   - raw predicted action
   - corrected action
   - probabilities
   - feature vector
   - elapsed times
   - heuristic diagnostics

Default oscillation mode:
    "strict_triplet"

Other supported modes:
    - "alternating_window"
    - "gated_alternating"

This script assumes:
- navigator.py provides NavigatorMLPInference
- the project root contains modules/ with:
    - modules.rgbd_similarity
    - modules.feature_based_point_cloud_registration

Typical integration:
- another controller acquires the current RGB-D frame
- another controller stores/manages the current key RGB-D frame
- this class only decides the action; it does NOT replace the keyframe itself
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import deque

import numpy as np
from PIL import Image
import torch
import quaternion


# ---------------------------------------------------------------------
# Project-root imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from navigator import NavigatorMLPInference  # noqa: E402
from modules.rgbd_similarity import RGBDSimilarity  # noqa: E402
from modules.feature_based_point_cloud_registration import FeatureBasedPointCloudRegistration  # noqa: E402


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "sim",
    "rmse",
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def load_rgb(path: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_depth(path: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(path))


def quaternion_to_angle_deg(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    Convert quaternion to absolute rotation angle in degrees.
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        return np.nan
    q = q / norm
    qw = np.clip(q[0], -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(np.abs(qw))
    return float(np.degrees(angle_rad))


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Real-world registration wrapper
# ---------------------------------------------------------------------
class RealFeatureBasedPointCloudRegistration(FeatureBasedPointCloudRegistration):
    """
    Same idea as in the notebook:
    source = current frame
    target = key frame

    compute_relative_pose(...) returns T_source_to_target, i.e.
    current/source -> key/target
    """

    def __init__(
        self,
        config: dict,
        device: str,
        id_run: int,
        feature_nav_conf: str = "LightGlue",
        feature_mode: str = "mnn",
        topological_map: bool = False,
        manual_operation: bool = False,
        fx: float = 610.1170,
        fy: float = 610.2250,
        cx: float = 323.7142,
        cy: float = 237.8927,
        depth_scale: float = 1e-3,
        invalid_depth_value: float = 0.0,
    ):
        super().__init__(
            config=config,
            device=device,
            id_run=id_run,
            feature_nav_conf=feature_nav_conf,
            feature_mode=feature_mode,
            topological_map=topological_map,
            manual_operation=manual_operation,
        )
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.depth_scale = float(depth_scale)
        self.invalid_depth_value = float(invalid_depth_value)

    def generate_pc_in_cam_ref_frame(self, depth_img: np.ndarray, T_cam_world=None) -> np.ndarray:
        if depth_img.ndim != 2:
            raise ValueError(f"depth_img must be HxW, got shape {depth_img.shape}")

        depth = depth_img.astype(np.float64) * self.depth_scale
        h, w = depth.shape

        u, v = np.meshgrid(
            np.arange(w, dtype=np.float64),
            np.arange(h, dtype=np.float64),
        )

        Z = depth
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        valid = np.isfinite(Z) & (Z > self.invalid_depth_value)
        X[~valid] = np.nan
        Y[~valid] = np.nan
        Z[~valid] = np.nan

        ones = np.ones_like(Z, dtype=np.float64)
        pc_cam_h = np.stack([X, Y, Z, ones], axis=0).reshape(4, -1)
        return pc_cam_h

    def get_ipc_from_pc(self, pc_cam: np.ndarray, kp_cam: np.ndarray, h: int, w: int) -> np.ndarray:
        kp = np.asarray(kp_cam, dtype=np.float64)
        xs = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)

        cam_key_id = ys * w + xs
        ipc_cam_h = pc_cam[:, cam_key_id]
        return ipc_cam_h[:3].T

    def _filter_valid_depth_correspondences(
        self,
        kp_source: np.ndarray,
        kp_target: np.ndarray,
        source_depth: np.ndarray,
        target_depth: np.ndarray,
    ):
        hs, ws = source_depth.shape[:2]
        ht, wt = target_depth.shape[:2]

        ks = np.asarray(kp_source, dtype=np.float64)
        kt = np.asarray(kp_target, dtype=np.float64)

        xs = np.clip(np.round(ks[:, 0]).astype(int), 0, ws - 1)
        ys = np.clip(np.round(ks[:, 1]).astype(int), 0, hs - 1)

        xt = np.clip(np.round(kt[:, 0]).astype(int), 0, wt - 1)
        yt = np.clip(np.round(kt[:, 1]).astype(int), 0, ht - 1)

        ds = source_depth[ys, xs]
        dt = target_depth[yt, xt]

        valid = (
            np.isfinite(ds)
            & np.isfinite(dt)
            & (ds > self.invalid_depth_value)
            & (dt > self.invalid_depth_value)
        )
        return ks[valid], kt[valid], int(len(ks)), int(np.sum(valid))

    def compute_relative_pose_with_debug(
        self,
        source_color: np.ndarray,
        source_depth: np.ndarray,
        target_color: np.ndarray,
        target_depth: np.ndarray,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        kp_source, kp_target = self.feature_nav.compute_matches(source_color, target_color)
        kp_source = np.asarray(kp_source)
        kp_target = np.asarray(kp_target)

        out: Dict[str, Any] = {
            "bot_lost": True,
            "rmse": np.nan,
            "t": None,
            "q": None,
            "T_source_to_target": None,
            "n_matches_raw": int(len(kp_source)),
            "n_matches_valid_depth": 0,
            "registration_success": False,
            "elapsed_ms_registration": np.nan,
        }

        if len(kp_source) < 4 or len(kp_target) < 4:
            out["elapsed_ms_registration"] = (time.perf_counter() - t0) * 1000.0
            return out

        kp_source, kp_target, n_raw, n_valid = self._filter_valid_depth_correspondences(
            kp_source, kp_target, source_depth, target_depth
        )
        out["n_matches_raw"] = n_raw
        out["n_matches_valid_depth"] = n_valid

        if len(kp_source) < 4 or len(kp_target) < 4:
            out["elapsed_ms_registration"] = (time.perf_counter() - t0) * 1000.0
            return out

        hs, ws = source_depth.shape[:2]
        ht, wt = target_depth.shape[:2]

        source_pc_h = self.generate_pc_in_cam_ref_frame(source_depth)
        target_pc_h = self.generate_pc_in_cam_ref_frame(target_depth)

        ipc_source = self.get_ipc_from_pc(source_pc_h, kp_source, hs, ws)
        ipc_target = self.get_ipc_from_pc(target_pc_h, kp_target, ht, wt)

        valid3d = np.isfinite(ipc_source).all(axis=1) & np.isfinite(ipc_target).all(axis=1)
        ipc_source = ipc_source[valid3d]
        ipc_target = ipc_target[valid3d]

        if len(ipc_source) < 4 or len(ipc_target) < 4:
            out["elapsed_ms_registration"] = (time.perf_counter() - t0) * 1000.0
            return out

        rmse, transformed_ipc_source, est_T_source_to_target = self.execute_SVD_registration(
            ipc_source, ipc_target
        )
        R = est_T_source_to_target[:3, :3]
        t = est_T_source_to_target[:3, 3]
        q = quaternion.from_rotation_matrix(R)

        out.update({
            "bot_lost": False,
            "rmse": float(rmse),
            "t": np.asarray(t, dtype=np.float64),
            "q": q,
            "T_source_to_target": est_T_source_to_target,
            "registration_success": True,
            "elapsed_ms_registration": (time.perf_counter() - t0) * 1000.0,
        })
        return out


# ---------------------------------------------------------------------
# Heuristic configuration
# ---------------------------------------------------------------------
@dataclass
class OscillationHeuristicConfig:
    mode: str = "strict_triplet"          # default = simplest mode
    history_len: int = 6
    cooldown_steps: int = 3

    # Used by alternating_window / gated_alternating
    alternating_window_size: int = 4
    min_alternations: int = 3

    # Used only by gated_alternating
    require_low_forward_confidence: bool = True
    max_forward_probability: float = 0.55

    require_no_recent_forward: bool = False
    recent_forward_window: int = 3

    require_small_progress: bool = False
    max_translation_norm: float = 0.40

    require_similarity_not_improving: bool = False
    similarity_window: int = 3
    min_similarity_improvement: float = 0.01


# ---------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------
@dataclass
class NavigatorDecision:
    step_index: int
    raw_action: str
    corrected_action: str
    heuristic_triggered: bool
    heuristic_reason: str

    probabilities: Dict[str, float]
    feature_vector: Dict[str, float]

    quality_info: Dict[str, Any] = field(default_factory=dict)
    timing_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Main autonomous loop class
# ---------------------------------------------------------------------
class AutonomousNavigatorLoop:
    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        label_encoder_path: Union[str, Path],
        device: Optional[str] = None,
        fx: float = 610.1170,
        fy: float = 610.2250,
        cx: float = 323.7142,
        cy: float = 237.8927,
        depth_scale: float = 1e-3,
        invalid_depth_value: float = 0.0,
        feature_nav_conf: str = "LightGlue",
        feature_mode: str = "mnn",
        heuristic_cfg: Optional[OscillationHeuristicConfig] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Inference model
        self.inference = NavigatorMLPInference(
            model_path=model_path,
            scaler_path=scaler_path,
            label_encoder_path=label_encoder_path,
            device=self.device,
        )

        # Feature extraction modules
        self.rgbd_similarity = RGBDSimilarity(
            device=self.device,
            threshold=0.95,
        )

        self.registration = RealFeatureBasedPointCloudRegistration(
            config={},
            device=self.device,
            id_run=0,
            feature_nav_conf=feature_nav_conf,
            feature_mode=feature_mode,
            topological_map=False,
            manual_operation=False,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            invalid_depth_value=invalid_depth_value,
        )

        # Heuristic config
        self.heuristic_cfg = heuristic_cfg or OscillationHeuristicConfig()

        # Internal history
        self.step_index = 0
        self.last_forced_update_step = -10**9

        self.raw_action_buffer: deque[str] = deque(maxlen=self.heuristic_cfg.history_len)
        self.corrected_action_buffer: deque[str] = deque(maxlen=self.heuristic_cfg.history_len)
        self.prob_buffer: deque[Dict[str, float]] = deque(maxlen=self.heuristic_cfg.history_len)
        self.feature_buffer: deque[Dict[str, float]] = deque(maxlen=self.heuristic_cfg.history_len)
        self.decision_history: List[NavigatorDecision] = []

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def reset(self) -> None:
        self.step_index = 0
        self.last_forced_update_step = -10**9
        self.raw_action_buffer.clear()
        self.corrected_action_buffer.clear()
        self.prob_buffer.clear()
        self.feature_buffer.clear()
        self.decision_history.clear()

    def get_history(self) -> List[NavigatorDecision]:
        return list(self.decision_history)

    def step(
        self,
        current_rgb: np.ndarray,
        current_depth: np.ndarray,
        key_rgb: np.ndarray,
        key_depth: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NavigatorDecision:
        """
        Process one navigator step.

        Inputs:
            current_rgb, current_depth = observed pair
            key_rgb, key_depth         = current key pair

        Output:
            NavigatorDecision with:
                - raw_action
                - corrected_action
                - elapsed times
                - heuristic diagnostics
                - probabilities
                - feature vector
        """
        metadata = metadata or {}
        t_total0 = time.perf_counter()

        # 1) Feature extraction
        features, quality_info, timing_ms = self._compute_feature_vector(
            current_rgb=current_rgb,
            current_depth=current_depth,
            key_rgb=key_rgb,
            key_depth=key_depth,
        )

        # 2) Raw MLP prediction
        t_infer0 = time.perf_counter()
        full_pred = self.inference.predict_full(features)
        timing_ms["inference_ms"] = (time.perf_counter() - t_infer0) * 1000.0

        raw_action = str(full_pred["predicted_label"])
        probabilities = dict(full_pred["probabilities"])

        # 3) Update history with raw info before heuristic
        self.raw_action_buffer.append(raw_action)
        self.prob_buffer.append(probabilities)
        self.feature_buffer.append(features)

        # 4) Heuristic correction
        corrected_action, heuristic_triggered, heuristic_reason = self._apply_loop_buffer_heuristic(
            raw_action=raw_action,
            probabilities=probabilities,
            features=features,
        )

        # 5) Save corrected output
        self.corrected_action_buffer.append(corrected_action)

        if heuristic_triggered and corrected_action == "update memory":
            self.last_forced_update_step = self.step_index

        timing_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0

        decision = NavigatorDecision(
            step_index=self.step_index,
            raw_action=raw_action,
            corrected_action=corrected_action,
            heuristic_triggered=heuristic_triggered,
            heuristic_reason=heuristic_reason,
            probabilities=probabilities,
            feature_vector=features,
            quality_info=quality_info,
            timing_ms=timing_ms,
            metadata=metadata,
        )

        self.decision_history.append(decision)
        self.step_index += 1
        return decision

    # -------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------
    def _compute_feature_vector(
        self,
        current_rgb: np.ndarray,
        current_depth: np.ndarray,
        key_rgb: np.ndarray,
        key_depth: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, float]]:
        """
        Compute the 9D feature vector required by the trained MLP:

        sim, rmse, x, y, z, qw, qx, qy, qz

        Convention:
            source = current frame
            target = key frame
        """
        timing_ms: Dict[str, float] = {}
        quality_info: Dict[str, Any] = {}

        # Similarity
        t0 = time.perf_counter()
        sim = self.rgbd_similarity.compute_image_similarity(
            current_rgb, current_depth, key_rgb, key_depth
        )
        timing_ms["similarity_ms"] = (time.perf_counter() - t0) * 1000.0

        # Registration with debug info
        reg_info = self.registration.compute_relative_pose_with_debug(
            source_color=current_rgb,
            source_depth=current_depth,
            target_color=key_rgb,
            target_depth=key_depth,
        )

        quality_info["registration_success"] = bool(reg_info["registration_success"])
        quality_info["bot_lost"] = bool(reg_info["bot_lost"])
        quality_info["n_matches_raw"] = int(reg_info["n_matches_raw"])
        quality_info["n_matches_valid_depth"] = int(reg_info["n_matches_valid_depth"])

        features = {
            "sim": safe_float(sim),
            "rmse": np.nan,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "qw": np.nan,
            "qx": np.nan,
            "qy": np.nan,
            "qz": np.nan,
        }

        timing_ms["registration_ms"] = safe_float(reg_info["elapsed_ms_registration"])

        if reg_info["registration_success"]:
            t = np.asarray(reg_info["t"], dtype=np.float64).reshape(-1)
            q = reg_info["q"]

            features["rmse"] = safe_float(reg_info["rmse"])
            if t.shape[0] >= 3:
                features["x"] = float(t[0])
                features["y"] = float(t[1])
                features["z"] = float(t[2])

            features["qw"] = float(q.w)
            features["qx"] = float(q.x)
            features["qy"] = float(q.y)
            features["qz"] = float(q.z)

        # Extra quality diagnostics not used by the model
        t_norm = np.nan
        if np.isfinite(features["x"]) and np.isfinite(features["y"]) and np.isfinite(features["z"]):
            t_norm = float(np.linalg.norm([features["x"], features["y"], features["z"]]))

        quality_info["translation_norm"] = t_norm
        quality_info["rotation_angle_deg"] = quaternion_to_angle_deg(
            features["qw"], features["qx"], features["qy"], features["qz"]
        )

        return features, quality_info, timing_ms

    # -------------------------------------------------------------
    # Oscillation heuristic dispatcher
    # -------------------------------------------------------------
    def _apply_loop_buffer_heuristic(
        self,
        raw_action: str,
        probabilities: Dict[str, float],
        features: Dict[str, float],
    ) -> Tuple[str, bool, str]:
        mode = self.heuristic_cfg.mode

        if mode == "strict_triplet":
            return self._heuristic_strict_triplet(raw_action, probabilities, features)

        if mode == "alternating_window":
            return self._heuristic_alternating_window(raw_action, probabilities, features)

        if mode == "gated_alternating":
            return self._heuristic_gated_alternating(raw_action, probabilities, features)

        raise ValueError(
            f"Unknown heuristic mode '{mode}'. "
            f"Supported: strict_triplet, alternating_window, gated_alternating"
        )

    # -------------------------------------------------------------
    # Heuristic mode 1: simplest default
    # -------------------------------------------------------------
    def _heuristic_strict_triplet(
        self,
        raw_action: str,
        probabilities: Dict[str, float],
        features: Dict[str, float],
    ) -> Tuple[str, bool, str]:
        """
        Simplest default mode.

        Trigger update memory if the last 3 RAW actions are exactly:
            [right, left, right]
            [left, right, left]

        with cooldown.
        """
        if not self._cooldown_ok():
            return raw_action, False, "cooldown_blocked"

        if len(self.raw_action_buffer) < 3:
            return raw_action, False, "insufficient_history"

        last3 = list(self.raw_action_buffer)[-3:]

        if last3 == ["right", "left", "right"]:
            return "update memory", True, "oscillation_rlr"

        if last3 == ["left", "right", "left"]:
            return "update memory", True, "oscillation_lrl"

        return raw_action, False, "none"

    # -------------------------------------------------------------
    # Heuristic mode 2: more robust oscillation window
    # -------------------------------------------------------------
    def _heuristic_alternating_window(
        self,
        raw_action: str,
        probabilities: Dict[str, float],
        features: Dict[str, float],
    ) -> Tuple[str, bool, str]:
        """
        Trigger update memory if the last N raw actions exhibit repeated
        left-right alternation.

        Example with N=4:
            [left, right, left, right]
            [right, left, right, left]

        More generally, count adjacent alternations within the window and
        trigger if there are enough.
        """
        if not self._cooldown_ok():
            return raw_action, False, "cooldown_blocked"

        n = int(self.heuristic_cfg.alternating_window_size)
        if len(self.raw_action_buffer) < n:
            return raw_action, False, "insufficient_history"

        window = list(self.raw_action_buffer)[-n:]

        # Only consider L/R oscillation windows
        if any(a not in ("left", "right") for a in window):
            return raw_action, False, "window_contains_non_turn"

        alternations = 0
        for a, b in zip(window[:-1], window[1:]):
            if a != b:
                alternations += 1

        if alternations >= int(self.heuristic_cfg.min_alternations):
            if window[0] == "left":
                return "update memory", True, "alternating_window_lrlr"
            return "update memory", True, "alternating_window_rlrl"

        return raw_action, False, "none"

    # -------------------------------------------------------------
    # Heuristic mode 3: gated oscillation
    # -------------------------------------------------------------
    def _heuristic_gated_alternating(
        self,
        raw_action: str,
        probabilities: Dict[str, float],
        features: Dict[str, float],
    ) -> Tuple[str, bool, str]:
        """
        Same oscillation idea as alternating_window, but requires extra gates.

        Possible gates:
        - low forward confidence
        - no recent forward in the raw action buffer
        - small translation norm
        - similarity not improving recently
        """
        corrected_action, triggered, reason = self._heuristic_alternating_window(
            raw_action, probabilities, features
        )
        if not triggered:
            return raw_action, False, reason

        cfg = self.heuristic_cfg

        # Gate 1: low forward confidence
        if cfg.require_low_forward_confidence:
            p_forward = float(probabilities.get("forward", 0.0))
            if p_forward > float(cfg.max_forward_probability):
                return raw_action, False, "gated_blocked_forward_confidence"

        # Gate 2: no recent forward
        if cfg.require_no_recent_forward:
            recent = list(self.raw_action_buffer)[-int(cfg.recent_forward_window):]
            if any(a == "forward" for a in recent):
                return raw_action, False, "gated_blocked_recent_forward"

        # Gate 3: small progress
        if cfg.require_small_progress:
            t_norm = safe_float(features.get("x", np.nan))
            x = safe_float(features.get("x", np.nan))
            y = safe_float(features.get("y", np.nan))
            z = safe_float(features.get("z", np.nan))
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                t_norm = float(np.linalg.norm([x, y, z]))
            else:
                t_norm = np.inf

            if t_norm > float(cfg.max_translation_norm):
                return raw_action, False, "gated_blocked_large_translation"

        # Gate 4: similarity not improving
        if cfg.require_similarity_not_improving:
            w = int(cfg.similarity_window)
            if len(self.feature_buffer) < w:
                return raw_action, False, "gated_blocked_insufficient_similarity_history"

            sims = [safe_float(f.get("sim", np.nan)) for f in list(self.feature_buffer)[-w:]]
            sims = [s for s in sims if np.isfinite(s)]
            if len(sims) >= 2:
                improvement = sims[-1] - sims[0]
                if improvement > float(cfg.min_similarity_improvement):
                    return raw_action, False, "gated_blocked_similarity_improving"

        return corrected_action, True, f"gated_{reason}"

    # -------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------
    def _cooldown_ok(self) -> bool:
        return (self.step_index - self.last_forced_update_step) > int(self.heuristic_cfg.cooldown_steps)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Example configuration
    # -----------------------------------------------------------------
    model_path = PROJECT_ROOT / "notebook_outputs" / "real_nav_lopo_weighted_ce" / "final_weighted_ce_mlp.pt"
    scaler_path = PROJECT_ROOT / "notebook_outputs" / "real_nav_lopo_weighted_ce" / "final_scaler.joblib"
    label_encoder_path = PROJECT_ROOT / "notebook_outputs" / "real_nav_lopo_weighted_ce" / "final_label_encoder.joblib"

    # Example heuristic modes:
    # 1) strict_triplet        -> simplest default
    # 2) alternating_window   -> repeated L/R alternation
    # 3) gated_alternating    -> alternation + extra conditions

    heuristic_cfg = OscillationHeuristicConfig(
        mode="strict_triplet",       # simplest default
        history_len=6,
        cooldown_steps=3,
        alternating_window_size=4,
        min_alternations=3,
        require_low_forward_confidence=True,
        max_forward_probability=0.55,
        require_no_recent_forward=False,
        require_small_progress=False,
        require_similarity_not_improving=False,
    )

    loop = AutonomousNavigatorLoop(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        device=None,
        fx=610.1170,
        fy=610.2250,
        cx=323.7142,
        cy=237.8927,
        depth_scale=1e-3,
        invalid_depth_value=0.0,
        feature_nav_conf="LightGlue",
        feature_mode="mnn",
        heuristic_cfg=heuristic_cfg,
    )

    # -----------------------------------------------------------------
    # Example with local files
    # Replace these paths with real current/key RGB-D files
    # -----------------------------------------------------------------
    current_rgb_path = PROJECT_ROOT / "example_data" / "current_rgb.png"
    current_depth_path = PROJECT_ROOT / "example_data" / "current_depth.png"
    key_rgb_path = PROJECT_ROOT / "example_data" / "key_rgb.png"
    key_depth_path = PROJECT_ROOT / "example_data" / "key_depth.png"

    if all(p.exists() for p in [current_rgb_path, current_depth_path, key_rgb_path, key_depth_path]):
        current_rgb = load_rgb(current_rgb_path)
        current_depth = load_depth(current_depth_path)
        key_rgb = load_rgb(key_rgb_path)
        key_depth = load_depth(key_depth_path)

        decision = loop.step(
            current_rgb=current_rgb,
            current_depth=current_depth,
            key_rgb=key_rgb,
            key_depth=key_depth,
            metadata={
                "current_rgb_path": str(current_rgb_path),
                "current_depth_path": str(current_depth_path),
                "key_rgb_path": str(key_rgb_path),
                "key_depth_path": str(key_depth_path),
                "episode_id": "demo_episode_001",
            },
        )

        print("\n=== Single decision ===")
        print("step_index         :", decision.step_index)
        print("raw_action         :", decision.raw_action)
        print("corrected_action   :", decision.corrected_action)
        print("heuristic_triggered:", decision.heuristic_triggered)
        print("heuristic_reason   :", decision.heuristic_reason)
        print("probabilities      :", decision.probabilities)
        print("feature_vector     :", decision.feature_vector)
        print("quality_info       :", decision.quality_info)
        print("timing_ms          :", decision.timing_ms)

    else:
        print("Example files not found. Update the example_data paths in __main__ to test locally.")
        print("The class is ready to be imported and used by the robot controller.")