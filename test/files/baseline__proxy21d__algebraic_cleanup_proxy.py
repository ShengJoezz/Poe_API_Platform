from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HIST_LEN = 32
SHORT_WIN = 8
MID_WIN = 16
LONG_WIN = 32
LOAD_WIN = 4


SUPerset_FEATURE_NAMES = (
    "bias",
    "RotSpeed",
    "BlPitchCMeas",
    "GenTqMeas",
    "HorWindV",
    "FA_Acc_TT",
    "VS_GenPwr_W",
    "sin_psi",
    "cos_psi",
    "sin_3psi",
    "cos_3psi",
    "FA_Acc_mean8_c",
    "FA_Acc_vel16_c",
    "FA_Acc_disp32_c",
    "BlPitchCMeas_c_lag01",
    "BlPitchCMeas_du_lag01",
    "PitOffsetApplied_c_lag01",
    "PitOffsetApplied_du_lag01",
    "HorWindV_mean8_c",
    "HorWindV_trend16_c",
    "dv_sin_psi",
    "dv_cos_psi",
    "dv_sin_3psi",
    "dv_cos_3psi",
    "dv2_c",
    "dv_domega",
    "dv_dbeta",
    "RootMOOPMean_mean4_c",
    "YawBrMyn_mean4_c",
    "RootMOOPMean_res4_c",
    "YawBrMyn_res4_c",
)


ROUTED_FEATURES: dict[str, tuple[str, ...]] = {
    "clean_global_fa": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_c_lag01",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
        "dv_domega",
        "dv_dbeta",
    ),
    "clean_global_twr": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_c_lag01",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
        "dv_domega",
        "dv_dbeta",
    ),
    "routed_fa": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_c_lag01",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
    ),
    "routed_twr": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_c_lag01",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
        "dv_domega",
        "dv_dbeta",
        "RootMOOPMean_res4_c",
        "YawBrMyn_res4_c",
    ),
    "routed_fa_nobetalag": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
    ),
    "routed_twr_nobetalag": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
        "dv_domega",
        "dv_dbeta",
        "RootMOOPMean_res4_c",
        "YawBrMyn_res4_c",
    ),
    "routed_fa_prunedwork": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "HorWindV",
        "FA_Acc_TT",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
    ),
    "routed_twr_prunedwork": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "HorWindV",
        "FA_Acc_TT",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "BlPitchCMeas_du_lag01",
        "PitOffsetApplied_c_lag01",
        "PitOffsetApplied_du_lag01",
        "HorWindV_trend16_c",
        "dv_sin_psi",
        "dv_cos_psi",
        "dv_sin_3psi",
        "dv_cos_3psi",
        "dv2_c",
        "dv_domega",
        "dv_dbeta",
        "RootMOOPMean_res4_c",
        "YawBrMyn_res4_c",
    ),
    "clean_global_power": (
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "VS_GenPwr_W",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
        "BlPitchCMeas_c_lag01",
        "PitOffsetApplied_c_lag01",
    ),
}


def feature_names_for_family_target(family: str, target: str) -> tuple[str, ...]:
    if family == "clean_global":
        if target == "FA_Acc_TT":
            return ROUTED_FEATURES["clean_global_fa"]
        if target == "TwrBsMyt":
            return ROUTED_FEATURES["clean_global_twr"]
        return ROUTED_FEATURES["clean_global_power"]
    if family == "target_routed_resload":
        if target == "FA_Acc_TT":
            return ROUTED_FEATURES["routed_fa"]
        if target == "TwrBsMyt":
            return ROUTED_FEATURES["routed_twr"]
        return ROUTED_FEATURES["clean_global_power"]
    if family == "target_routed_nobetalag":
        if target == "FA_Acc_TT":
            return ROUTED_FEATURES["routed_fa_nobetalag"]
        if target == "TwrBsMyt":
            return ROUTED_FEATURES["routed_twr_nobetalag"]
        return ROUTED_FEATURES["clean_global_power"]
    if family == "target_routed_prunedwork":
        if target == "FA_Acc_TT":
            return ROUTED_FEATURES["routed_fa_prunedwork"]
        if target == "TwrBsMyt":
            return ROUTED_FEATURES["routed_twr_prunedwork"]
        return ROUTED_FEATURES["clean_global_power"]
    if family == "deploy_acc_prunedwork":
        return ROUTED_FEATURES["routed_fa_prunedwork"]
    raise ValueError(f"unsupported family={family!r}")


def is_algebraic_cleanup_proxy_mode(feature_mode: str) -> bool:
    return str(feature_mode) == "algebraic_cleanup_proxy"


def make_algebraic_cleanup_params() -> dict:
    return {
        "kind": "algebraic_cleanup_proxy",
        "hist_len": int(HIST_LEN),
        "short_win": int(SHORT_WIN),
        "mid_win": int(MID_WIN),
        "long_win": int(LONG_WIN),
        "load_win": int(LOAD_WIN),
    }


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _root_mean_from_row(meas: Mapping[str, object]) -> float:
    vals = [
        _safe_float(meas.get("rootMOOP(1)", float("nan")), float("nan")),
        _safe_float(meas.get("rootMOOP(2)", float("nan")), float("nan")),
        _safe_float(meas.get("rootMOOP(3)", float("nan")), float("nan")),
    ]
    arr = np.asarray(vals, dtype=float)
    if np.isfinite(arr).sum() == 0:
        return 0.0
    return float(np.nanmean(arr))


def _make_history(length: int, value: float) -> deque[float]:
    hist: deque[float] = deque(maxlen=max(int(length), 1))
    for _ in range(hist.maxlen):
        hist.append(float(value))
    return hist


def _lag_value(hist: deque[float], lag: int) -> float:
    lag = int(lag)
    if lag <= 0:
        return float(hist[-1])
    if len(hist) < lag:
        return float(hist[0])
    return float(list(hist)[-lag])


def _tail_array(hist: deque[float], n: int) -> np.ndarray:
    arr = np.asarray(hist, dtype=float)
    if arr.size <= int(n):
        return arr
    return arr[-int(n) :]


def _mean_centered(arr: np.ndarray, *, ref: float, scale: float) -> float:
    return float((np.mean(arr) - ref) / max(scale, 1e-12))


def _integral_proxy(arr: np.ndarray, *, ref: float, scale: float, ts: float) -> float:
    centered = (np.asarray(arr, dtype=float) - float(ref)) / max(float(scale), 1e-12)
    return float(np.sum(centered) * float(ts))


def _double_integral_proxy(arr: np.ndarray, *, ref: float, scale: float, ts: float) -> float:
    centered = (np.asarray(arr, dtype=float) - float(ref)) / max(float(scale), 1e-12)
    vel = np.cumsum(centered) * float(ts)
    return float(np.sum(vel) * float(ts))


@dataclass
class AlgebraicCleanupRuntime:
    ts: float
    refs: Mapping[str, float]
    params: Mapping[str, object]
    load_proj: Mapping[str, object] | None = None
    acc_hist: deque[float] | None = None
    pitch_hist: deque[float] | None = None
    cmd_hist: deque[float] | None = None
    wind_hist: deque[float] | None = None
    root_hist: deque[float] | None = None
    yaw_hist: deque[float] | None = None
    initialized: bool = False

    def _initialize(self, *, acc: float, beta: float, u_prev: float, wind: float, root_mean: float, yaw: float) -> None:
        hist_len = int(self.params["hist_len"])
        self.acc_hist = _make_history(hist_len, acc)
        self.pitch_hist = _make_history(2, beta)
        self.cmd_hist = _make_history(2, u_prev)
        self.wind_hist = _make_history(hist_len, wind)
        self.root_hist = _make_history(int(self.params["load_win"]), root_mean)
        self.yaw_hist = _make_history(int(self.params["load_win"]), yaw)
        self.initialized = True

    def update(self, meas: Mapping[str, object], *, u_prev: float = 0.0) -> dict[str, float]:
        beta = _safe_float(meas.get("BlPitchCMeas", 0.0), 0.0)
        wind = _safe_float(meas.get("HorWindV", 0.0), 0.0)
        acc = _safe_float(meas.get("FA_Acc_TT", 0.0), 0.0)
        yaw = _safe_float(meas.get("YawBrMyn", 0.0), 0.0)
        root_mean = _root_mean_from_row(meas)
        u_prev = _safe_float(u_prev, 0.0)
        if not self.initialized:
            self._initialize(acc=acc, beta=beta, u_prev=u_prev, wind=wind, root_mean=root_mean, yaw=yaw)
        else:
            assert self.acc_hist is not None
            assert self.pitch_hist is not None
            assert self.cmd_hist is not None
            assert self.wind_hist is not None
            assert self.root_hist is not None
            assert self.yaw_hist is not None
            self.acc_hist.append(acc)
            self.pitch_hist.append(beta)
            self.cmd_hist.append(u_prev)
            self.wind_hist.append(wind)
            self.root_hist.append(root_mean)
            self.yaw_hist.append(yaw)

        assert self.acc_hist is not None
        assert self.pitch_hist is not None
        assert self.cmd_hist is not None
        assert self.wind_hist is not None
        assert self.root_hist is not None
        assert self.yaw_hist is not None

        omega = _safe_float(meas.get("RotSpeed", 0.0), 0.0)
        gen_tq = _safe_float(meas.get("GenTqMeas", 0.0), 0.0)
        az = _safe_float(meas.get("Azimuth", 0.0), 0.0)
        sin_psi = math.sin(az)
        cos_psi = math.cos(az)
        sin_3psi = math.sin(3.0 * az)
        cos_3psi = math.cos(3.0 * az)

        omega_ref = float(self.refs["omega_ref"])
        omega_scale = float(self.refs["omega_scale"])
        beta_ref = float(self.refs["beta_ref"])
        beta_scale = float(self.refs["beta_scale"])
        wind_ref = float(self.refs["wind_ref"])
        wind_scale = float(self.refs["wind_scale"])
        acc_ref = float(self.refs["acc_ref"])
        acc_scale = float(self.refs["acc_scale"])
        cmd_ref = float(self.refs["cmd_ref"])
        cmd_scale = float(self.refs["cmd_scale"])
        root_ref = float(self.refs["root_ref"])
        root_scale = float(self.refs["root_scale"])
        yaw_ref = float(self.refs["yaw_ref"])
        yaw_scale = float(self.refs["yaw_scale"])

        wind_short = _tail_array(self.wind_hist, int(self.params["short_win"]))
        acc_short = _tail_array(self.acc_hist, int(self.params["short_win"]))
        acc_mid = _tail_array(self.acc_hist, int(self.params["mid_win"]))
        acc_long = _tail_array(self.acc_hist, int(self.params["long_win"]))
        root_arr = _tail_array(self.root_hist, int(self.params["load_win"]))
        yaw_arr = _tail_array(self.yaw_hist, int(self.params["load_win"]))

        wind_mean8_raw = float(np.mean(wind_short))
        dv_local = (wind - wind_mean8_raw) / max(wind_scale, 1e-12)
        domega = (omega - omega_ref) / max(omega_scale, 1e-12)
        dbeta = (beta - beta_ref) / max(beta_scale, 1e-12)

        out = {str(k): _safe_float(v, float("nan")) for k, v in meas.items()}
        out.update(
            {
                "bias": 1.0,
                "RotSpeed": float(omega),
                "BlPitchCMeas": float(beta),
                "GenTqMeas": float(gen_tq),
                "HorWindV": float(wind),
                "FA_Acc_TT": float(acc),
                "VS_GenPwr_W": _safe_float(meas.get("VS_GenPwr", float("nan")), float("nan")),
                "sin_psi": float(sin_psi),
                "cos_psi": float(cos_psi),
                "sin_3psi": float(sin_3psi),
                "cos_3psi": float(cos_3psi),
                "FA_Acc_mean8_c": _mean_centered(acc_short, ref=acc_ref, scale=acc_scale),
                "FA_Acc_vel16_c": _integral_proxy(acc_mid, ref=acc_ref, scale=acc_scale, ts=self.ts),
                "FA_Acc_disp32_c": _double_integral_proxy(acc_long, ref=acc_ref, scale=acc_scale, ts=self.ts),
                "BlPitchCMeas_c_lag01": float((_lag_value(self.pitch_hist, 1) - beta_ref) / max(beta_scale, 1e-12)),
                "BlPitchCMeas_du_lag01": float(_lag_value(self.pitch_hist, 1) - _lag_value(self.pitch_hist, 2)),
                "PitOffsetApplied_c_lag01": float((_lag_value(self.cmd_hist, 1) - cmd_ref) / max(cmd_scale, 1e-12)),
                "PitOffsetApplied_du_lag01": float(_lag_value(self.cmd_hist, 1) - _lag_value(self.cmd_hist, 2)),
                "HorWindV_mean8_c": float((wind_mean8_raw - wind_ref) / max(wind_scale, 1e-12)),
                "HorWindV_trend16_c": float((_lag_value(self.wind_hist, 1) - _lag_value(self.wind_hist, int(self.params["mid_win"]))) / max(wind_scale, 1e-12)),
                "dv_sin_psi": float(dv_local * sin_psi),
                "dv_cos_psi": float(dv_local * cos_psi),
                "dv_sin_3psi": float(dv_local * sin_3psi),
                "dv_cos_3psi": float(dv_local * cos_3psi),
                "dv2_c": float(dv_local * dv_local),
                "dv_domega": float(dv_local * domega),
                "dv_dbeta": float(dv_local * dbeta),
                "RootMOOPMean_mean4_c": float((np.mean(root_arr) - root_ref) / max(root_scale, 1e-12)),
                "YawBrMyn_mean4_c": float((np.mean(yaw_arr) - yaw_ref) / max(yaw_scale, 1e-12)),
            }
        )
        out["RootMOOPMean_res4_c"] = out["RootMOOPMean_mean4_c"]
        out["YawBrMyn_res4_c"] = out["YawBrMyn_mean4_c"]
        if self.load_proj:
            pred_cols = list(self.load_proj.get("predictor_cols", ()))
            pred_vec = np.asarray([out.get(c, 0.0) for c in pred_cols], dtype=float)
            for raw_name, res_name in (("RootMOOPMean_mean4_c", "RootMOOPMean_res4_c"), ("YawBrMyn_mean4_c", "YawBrMyn_res4_c")):
                beta = np.asarray(self.load_proj.get(raw_name, []), dtype=float)
                if beta.size == pred_vec.size and beta.size > 0:
                    out[res_name] = float(out[raw_name] - float(pred_vec @ beta))
        return out


def compute_algebraic_cleanup_refs(dfs: Iterable[pd.DataFrame], *, trim_start_s: float, ts: float) -> dict:
    i0 = int(math.floor(float(trim_start_s) / float(ts)))
    omega_list: list[np.ndarray] = []
    beta_list: list[np.ndarray] = []
    wind_list: list[np.ndarray] = []
    acc_list: list[np.ndarray] = []
    cmd_list: list[np.ndarray] = []
    root_list: list[np.ndarray] = []
    yaw_list: list[np.ndarray] = []
    for df in dfs:
        if len(df) <= i0 + 2:
            continue
        omega_list.append(df["RotSpeed"].to_numpy(dtype=float)[i0:])
        beta_list.append(df["BlPitchCMeas"].to_numpy(dtype=float)[i0:])
        wind_list.append(df["HorWindV"].to_numpy(dtype=float)[i0:])
        acc_list.append(df["FA_Acc_TT"].to_numpy(dtype=float)[i0:])
        if "pit_offset_rad" in df.columns:
            cmd_list.append(df["pit_offset_rad"].to_numpy(dtype=float)[i0:])
        root_cols = [c for c in ("rootMOOP(1)", "rootMOOP(2)", "rootMOOP(3)") if c in df.columns]
        if root_cols:
            root_list.append(df[root_cols].to_numpy(dtype=float).mean(axis=1)[i0:])
        if "YawBrMyn" in df.columns:
            yaw_list.append(df["YawBrMyn"].to_numpy(dtype=float)[i0:])

    def _cat(vals: list[np.ndarray], default_mean: float, default_scale: float) -> tuple[float, float]:
        if not vals:
            return float(default_mean), float(default_scale)
        arr = np.concatenate(vals)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default_mean), float(default_scale)
        mean = float(np.mean(arr))
        scale = float(np.std(arr))
        if not math.isfinite(scale) or scale < 1e-9:
            scale = float(default_scale)
        return mean, scale

    omega_ref, omega_scale = _cat(omega_list, 1.0, 1.0)
    beta_ref, beta_scale = _cat(beta_list, 1.0, 1.0)
    wind_ref, wind_scale = _cat(wind_list, 1.0, 1.0)
    acc_ref, acc_scale = _cat(acc_list, 0.0, 1.0)
    cmd_ref, cmd_scale = _cat(cmd_list, 0.0, 1.0)
    root_ref, root_scale = _cat(root_list, 0.0, 1.0)
    yaw_ref, yaw_scale = _cat(yaw_list, 0.0, 1.0)
    if abs(omega_ref) <= 1e-6:
        omega_ref = 1.0
    if abs(beta_ref) <= 1e-6:
        beta_ref = 1.0
    if abs(wind_ref) <= 1e-6:
        wind_ref = 1.0
    return {
        "omega_ref": float(omega_ref),
        "omega_scale": float(omega_scale),
        "beta_ref": float(beta_ref),
        "beta_scale": float(beta_scale),
        "wind_ref": float(wind_ref),
        "wind_scale": float(wind_scale),
        "acc_ref": float(acc_ref),
        "acc_scale": float(acc_scale),
        "cmd_ref": float(cmd_ref),
        "cmd_scale": float(cmd_scale),
        "root_ref": float(root_ref),
        "root_scale": float(root_scale),
        "yaw_ref": float(yaw_ref),
        "yaw_scale": float(yaw_scale),
    }


def build_load_projection(aug_df: pd.DataFrame) -> dict:
    predictor_cols = [
        "bias",
        "RotSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "FA_Acc_mean8_c",
        "FA_Acc_vel16_c",
        "FA_Acc_disp32_c",
        "HorWindV_mean8_c",
        "HorWindV_trend16_c",
    ]
    X = aug_df.loc[:, predictor_cols].to_numpy(dtype=float)
    out = {"predictor_cols": predictor_cols}
    for raw_name in ("RootMOOPMean_mean4_c", "YawBrMyn_mean4_c"):
        z = aug_df[raw_name].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        out[raw_name] = beta.tolist()
    return out


def simulate_algebraic_cleanup_df(
    df: pd.DataFrame,
    *,
    refs: Mapping[str, float],
    params: Mapping[str, object],
    ts: float,
    load_proj: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    runtime = AlgebraicCleanupRuntime(ts=float(ts), refs=refs, params=params, load_proj=load_proj)
    rows: list[dict[str, float]] = []
    for row in df.to_dict(orient="records"):
        cur = runtime.update(row, u_prev=_safe_float(row.get("pit_offset_rad", 0.0), 0.0))
        rows.append(cur)
    return pd.DataFrame(rows)
