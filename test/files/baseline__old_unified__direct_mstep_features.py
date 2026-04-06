from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_MODE_CHOICES = (
    "base",
    "plus_yawbr",
    "no_pitch",
    "no_pitch_plus_yawbr",
    "phys_coupled",
    "koopman23_proxy",
    "state_full_proxy",
)


def u_blocks_for_mode(u_mode: str) -> int:
    if u_mode == "fixed":
        return 1
    if u_mode in ("tv_wind2", "tv_wind2_centered"):
        return 2
    if u_mode in ("tv_omega_beta_wind2", "tv_omega_beta_wind2_centered"):
        return 4
    raise ValueError(f"unknown u_mode: {u_mode}")


def _feature_arrays_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    az = df["Azimuth"].to_numpy(dtype=float)
    omega = df["RotSpeed"].to_numpy(dtype=float)
    beta = df["BlPitchCMeas"].to_numpy(dtype=float)
    wind = df["HorWindV"].to_numpy(dtype=float)
    acc = df["FA_Acc_TT"].to_numpy(dtype=float)
    gen_tq = df["GenTqMeas"].to_numpy(dtype=float)

    mapping: Dict[str, np.ndarray] = {
        "bias": np.ones(len(df), dtype=float),
        "RotSpeed": omega,
        "GenSpeed": df["GenSpeed"].to_numpy(dtype=float),
        "BlPitchCMeas": beta,
        "GenTqMeas": gen_tq,
        "HorWindV": wind,
        "FA_Acc_TT": acc,
        "VS_GenPwr_W": df["VS_GenPwr"].to_numpy(dtype=float),
        "sin_psi": np.sin(az),
        "cos_psi": np.cos(az),
        "sin_3psi": np.sin(3.0 * az),
        "cos_3psi": np.cos(3.0 * az),
        "HorWindV2": wind * wind,
        "HorWindV_RotSpeed": wind * omega,
        "HorWindV_BlPitchCMeas": wind * beta,
        "HorWindV_FA_Acc_TT": wind * acc,
        "FA_AccF": df["FA_AccF"].to_numpy(dtype=float) if "FA_AccF" in df.columns else np.full(len(df), np.nan, dtype=float),
        "NacIMU_FA_RAcc": df["NacIMU_FA_RAcc"].to_numpy(dtype=float) if "NacIMU_FA_RAcc" in df.columns else np.full(len(df), np.nan, dtype=float),
        "FA_Acc_TT2": acc * acc,
        "RotSpeed_GenTqMeas": omega * gen_tq,
        "RotSpeed2": omega * omega,
        "RotSpeed_BlPitchCMeas": omega * beta,
        "BlPitchCMeas2": beta * beta,
        "YawBrMyn": df["YawBrMyn"].to_numpy(dtype=float) if "YawBrMyn" in df.columns else np.full(len(df), np.nan, dtype=float),
    }
    for name in df.columns:
        if name in mapping:
            continue
        try:
            arr = df[name].to_numpy(dtype=float)
        except Exception:
            continue
        if arr.shape == (len(df),):
            mapping[name] = arr
    return mapping


def _feature_scalars_from_meas(meas: Dict[str, float]) -> Dict[str, float]:
    az = float(meas["Azimuth"])
    omega = float(meas["RotSpeed"])
    beta = float(meas["BlPitchCMeas"])
    wind = float(meas["HorWindV"])
    acc = float(meas.get("FA_Acc_TT", float("nan")))
    gen_tq = float(meas["GenTqMeas"])

    mapping = {
        "bias": 1.0,
        "RotSpeed": omega,
        "GenSpeed": float(meas["GenSpeed"]),
        "BlPitchCMeas": beta,
        "GenTqMeas": gen_tq,
        "HorWindV": wind,
        "FA_Acc_TT": acc,
        "VS_GenPwr_W": float(meas["VS_GenPwr"]),
        "sin_psi": float(np.sin(az)),
        "cos_psi": float(np.cos(az)),
        "sin_3psi": float(np.sin(3.0 * az)),
        "cos_3psi": float(np.cos(3.0 * az)),
        "HorWindV2": wind * wind,
        "HorWindV_RotSpeed": wind * omega,
        "HorWindV_BlPitchCMeas": wind * beta,
        "HorWindV_FA_Acc_TT": wind * acc,
        "FA_AccF": float(meas.get("FA_AccF", float("nan"))),
        "NacIMU_FA_RAcc": float(meas.get("NacIMU_FA_RAcc", float("nan"))),
        "FA_Acc_TT2": acc * acc,
        "RotSpeed_GenTqMeas": omega * gen_tq,
        "RotSpeed2": omega * omega,
        "RotSpeed_BlPitchCMeas": omega * beta,
        "BlPitchCMeas2": beta * beta,
        "YawBrMyn": float(meas.get("YawBrMyn", float("nan"))),
    }
    for name, value in meas.items():
        try:
            v = float(value)
        except Exception:
            continue
        mapping[str(name)] = v
    return mapping


def feature_names_for_mode(feature_mode: str) -> List[str]:
    base = [
        "bias",
        "RotSpeed",
        "GenSpeed",
        "BlPitchCMeas",
        "GenTqMeas",
        "HorWindV",
        "FA_Acc_TT",
        "VS_GenPwr_W",
        "sin_psi",
        "cos_psi",
        "sin_3psi",
        "cos_3psi",
    ]
    if feature_mode == "base":
        return list(base)
    if feature_mode == "plus_yawbr":
        return list(base) + ["YawBrMyn"]
    if feature_mode == "no_pitch":
        return [n for n in base if n != "BlPitchCMeas"]
    if feature_mode == "no_pitch_plus_yawbr":
        return [n for n in base if n != "BlPitchCMeas"] + ["YawBrMyn"]
    if feature_mode == "phys_coupled":
        return list(base) + [
            "HorWindV2",
            "HorWindV_RotSpeed",
            "HorWindV_BlPitchCMeas",
            "HorWindV_FA_Acc_TT",
        ]
    if feature_mode == "koopman23_proxy":
        return list(base) + [
            "HorWindV2",
            "HorWindV_RotSpeed",
            "HorWindV_BlPitchCMeas",
            "HorWindV_FA_Acc_TT",
            "FA_AccF",
            "NacIMU_FA_RAcc",
            "FA_Acc_TT2",
            "RotSpeed_GenTqMeas",
            "RotSpeed2",
            "RotSpeed_BlPitchCMeas",
            "BlPitchCMeas2",
        ]
    if feature_mode == "state_full_proxy":
        return [
            "bias",
            "RotSpeed",
            "GenSpeed",
            "BlPitchCMeas",
            "GenTqMeas",
            "HorWindV",
            "FA_Acc_TT",
            "VS_GenPwr_W",
            "sin_psi",
            "cos_psi",
            "sin_3psi",
            "cos_3psi",
            "HorWindV2",
            "HorWindV_RotSpeed",
            "HorWindV_BlPitchCMeas",
            "HorWindV_FA_Acc_TT",
            "TowerBendProxy",
            "TowerBendProxy_lp",
            "dTowerBendProxy_lp_dt",
            "TowerDispObs_c",
            "TowerVelObs_c",
            "dv_sin_psi",
            "dv_cos_psi",
            "dv_sin_3psi",
            "dv_cos_3psi",
            "RotSpeed_TowerVelObs",
            "BlPitchCMeas_TowerVelObs",
            "HorWindV_TowerVelObs",
            "domega_dbeta",
            "dv_dbeta",
            "RotSpeed_BlPitchCMeas",
            "HorWindV_RotSpeed_BlPitchCMeas",
            "Pitch_lp",
            "Pitch_lag_err",
            "Wind_lp",
            "Wind_lag_err",
        ]
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def features_from_meas_df(df: pd.DataFrame, *, feature_mode: str) -> Tuple[np.ndarray, List[str]]:
    mapping = _feature_arrays_from_df(df)
    names = feature_names_for_mode(feature_mode)
    cols: List[np.ndarray] = []
    for name in names:
        arr = mapping[name]
        if np.isnan(arr).any():
            raise KeyError(f"feature_mode={feature_mode} requires meas.csv column(s) for '{name}'")
        cols.append(arr)
    return np.column_stack(cols), names


def feature_vector_from_meas(meas: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    mapping = _feature_scalars_from_meas(meas)
    vals: List[float] = []
    for name in feature_names:
        val = mapping[name]
        if not np.isfinite(val):
            raise KeyError(f"measurement packet missing required feature '{name}'")
        vals.append(float(val))
    return np.asarray(vals, dtype=float)


def u_features_from_sequences(
    *,
    u_mode: str,
    u_sched_ref: dict | None,
    uk: np.ndarray,
    omega0: np.ndarray,
    beta0: np.ndarray,
    wind0: np.ndarray,
) -> np.ndarray:
    if u_mode == "fixed":
        return uk
    if u_mode == "tv_wind2":
        v2 = wind0 * wind0
        return np.column_stack([uk, uk * v2[:, None]])
    if u_mode == "tv_omega_beta_wind2":
        v2 = wind0 * wind0
        return np.column_stack([uk, uk * omega0[:, None], uk * beta0[:, None], uk * v2[:, None]])
    if u_mode == "tv_wind2_centered":
        if u_sched_ref is None:
            raise ValueError("u_mode=tv_wind2_centered requires u_sched_ref")
        v2 = wind0 * wind0
        v2_ref = float(u_sched_ref["wind_ref"]) ** 2
        dv2 = (v2 - v2_ref) / v2_ref
        return np.column_stack([uk, uk * dv2[:, None]])
    if u_mode == "tv_omega_beta_wind2_centered":
        if u_sched_ref is None:
            raise ValueError("u_mode=tv_omega_beta_wind2_centered requires u_sched_ref")
        domega = (omega0 - float(u_sched_ref["omega_ref"])) / float(u_sched_ref["omega_ref"])
        dbeta = (beta0 - float(u_sched_ref["beta_ref"])) / float(u_sched_ref["beta_ref"])
        v2 = wind0 * wind0
        v2_ref = float(u_sched_ref["wind_ref"]) ** 2
        dv2 = (v2 - v2_ref) / v2_ref
        return np.column_stack([uk, uk * domega[:, None], uk * dbeta[:, None], uk * dv2[:, None]])
    raise ValueError(f"unknown u_mode: {u_mode}")
