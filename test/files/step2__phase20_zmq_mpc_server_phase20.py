#!/usr/bin/env python3
"""
ROSCO ZeroMQ (WFC) server that runs a convex QP-MPC and returns ZMQ_PitOffset.

This provides "true closed-loop" enhancement-layer MPC without MATLAB/Simulink:
  - OpenFAST runs ROSCO (DISCON) with ZMQ_Mode enabled
  - ROSCO's ZMQ client sends the measurement vector defined by wfc_interface.yaml
  - This server returns 8 setpoints; we only modify ZMQ_PitOffset(1..3)

MPC model:
  Direct multi-step linear predictors trained by `scripts/stage4_zmq_train_direct_mstep.py`.

Decision variable:
  u = [u0, u1, ..., u_{N-1}]  (collective pitch offset sequence, radians)
  The applied command is u0 (receding horizon).

Objective:
  Minimize band-pass energy of a predicted load proxy y (FA_Acc_TT or TwrBsMyt),
  plus quadratic penalties on u and du.

Constraints:
  - |u| <= u_max
  - |du| <= du_max   (du0 uses previous applied u_prev)
  - Optional DC / low-frequency projection equality constraints on u
  - Optional mean power lower-bound constraint using predicted GenPwr

All computations remain linear/convex (QP) and use OSQP.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import zmq
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, firwin, hilbert

import osqp
import scipy.sparse as sp

def _find_repo_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "wind_mpc_lab").is_dir():
            return cand
    raise FileNotFoundError(f"failed to locate repo root above {start}")


_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_FILE_DIR)
for _p in (_FILE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from wind_mpc_lab.direct_mstep_features import feature_vector_from_meas
from wind_mpc_lab.state_full_proxy import StateFullProxyRuntime
try:
    from algebraic_cleanup_proxy import AlgebraicCleanupRuntime
except Exception:
    # The Phase 3 branch carries only a thin proxy wrapper locally.
    import delta_u_proxy as _delta_u_proxy  # type: ignore

    AlgebraicCleanupRuntime = _delta_u_proxy._base.AlgebraicCleanupRuntime
try:
    from delay_rich_proxy import DelayRichProxyRuntime
except Exception:
    DelayRichProxyRuntime = None  # type: ignore[assignment]


@dataclass(frozen=True)
class InterfaceDef:
    measurements: list[str]
    setpoints: list[str]
    setpoints_default: list[float]


def _load_interface(path: Path) -> InterfaceDef:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    meas = list(data["measurements"])
    setp = list(data["setpoints"])
    defaults = [float(x) for x in data["setpoints_default"]]
    if len(setp) != 8 or len(defaults) != 8:
        raise ValueError(f"Expected 8 setpoints/defaults, got {len(setp)}/{len(defaults)}")
    return InterfaceDef(measurements=meas, setpoints=setp, setpoints_default=defaults)


def _parse_floats(msg: str) -> list[float]:
    msg = msg.replace("\x00", "")
    out: list[float] = []
    for tok in msg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _load_model(models_dir: Path, *, wind_mps: int, target: str) -> tuple[dict, list[np.ndarray]]:
    meta = json.loads((models_dir / f"direct_ws{wind_mps:02d}_{target}.json").read_text(encoding="utf-8"))
    npz = np.load(models_dir / f"direct_ws{wind_mps:02d}_{target}.npz", allow_pickle=False)
    N = int(meta["meta"]["N"])
    thetas = [npz[f"theta_p{p:02d}"] for p in range(1, N + 1)]
    return meta, thetas


def _maybe_load_model(models_dir: Optional[Path], *, wind_mps: int, target: str) -> tuple[dict, list[np.ndarray]] | None:
    if models_dir is None:
        return None
    json_path = models_dir / f"direct_ws{wind_mps:02d}_{target}.json"
    npz_path = models_dir / f"direct_ws{wind_mps:02d}_{target}.npz"
    if not json_path.exists() or not npz_path.exists():
        return None
    return _load_model(models_dir, wind_mps=wind_mps, target=target)


def _feature_vector(meas: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    return feature_vector_from_meas(meas, feature_names)


def _make_feature_runtime(model_meta: dict, *, ts: float) -> StateFullProxyRuntime | AlgebraicCleanupRuntime | object | None:
    feature_mode = str(model_meta.get("feature_mode", "base"))
    if feature_mode not in ("state_full_proxy", "delay_rich_proxy", "algebraic_cleanup_proxy"):
        return None
    runtime_meta = model_meta.get("feature_runtime", {})
    if not isinstance(runtime_meta, dict):
        raise ValueError(f"{feature_mode} requires feature_runtime metadata")
    refs = runtime_meta.get("refs", {})
    params = runtime_meta.get("params", {})
    if not isinstance(refs, dict) or not isinstance(params, dict):
        raise ValueError(f"invalid {feature_mode} feature_runtime metadata")
    if feature_mode == "state_full_proxy":
        return StateFullProxyRuntime(ts=float(ts), refs=refs, params=params)
    if feature_mode == "algebraic_cleanup_proxy":
        load_proj = runtime_meta.get("load_proj", {})
        if not isinstance(load_proj, dict):
            raise ValueError("invalid algebraic_cleanup_proxy feature_runtime load_proj metadata")
        return AlgebraicCleanupRuntime(ts=float(ts), refs=refs, params=params, load_proj=load_proj)
    if DelayRichProxyRuntime is None:
        raise ValueError("delay_rich_proxy runtime requested but module is unavailable in this experiment branch")
    return DelayRichProxyRuntime(ts=float(ts), refs=refs, params=params)


def _safe_float(x: object, default: float) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return float(val)


def _objective_to_target(objective: str) -> str:
    objective = str(objective)
    if objective == "acc_bp":
        return "FA_Acc_TT"
    if objective == "myt_bp":
        return "TwrBsMyt"
    if objective == "tower_env_l2":
        return "TowerBendEnv"
    if objective == "tower_proxy_bp":
        return "TowerProxyEq"
    if objective == "tower_latent_bp":
        return "TowerLatentPC1"
    if objective == "tower_latent2_bp":
        return "TowerLatentPC2"
    if objective == "yaw_bp":
        return "YawBrMyn"
    raise ValueError(f"unknown objective: {objective!r}")


def _lp1(prev: float, meas: float, *, ts: float, tau_s: float) -> float:
    meas = float(meas)
    if not math.isfinite(meas):
        return float(prev)
    tau_s = float(tau_s)
    if (not math.isfinite(tau_s)) or tau_s <= 0.0:
        return meas
    alpha = math.exp(-float(ts) / tau_s)
    return float(alpha * float(prev) + (1.0 - alpha) * meas)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _parse_schedule_triplets(spec: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    text = str(spec).strip()
    if not text:
        return out
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        toks = [t.strip() for t in raw.split(",")]
        if len(toks) != 3:
            raise ValueError(f"invalid schedule triplet {raw!r}; expected 'v,beta_deg,value'")
        out.append((float(toks[0]), float(toks[1]), float(toks[2])))
    return out


def _piecewise_interp(
    *,
    x: float,
    points: list[tuple[float, float]],
    log_interp: bool,
    default_value: float,
) -> float:
    if not points:
        return float(default_value)
    pts = sorted(((float(px), float(py)) for px, py in points), key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) == 1:
        return float(ys[0])
    x = float(x)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            dx = max(float(x1 - x0), 1e-12)
            alpha = float((x - x0) / dx)
            if bool(log_interp) and y0 > 0.0 and y1 > 0.0:
                return float(math.exp((1.0 - alpha) * math.log(y0) + alpha * math.log(y1)))
            return float((1.0 - alpha) * y0 + alpha * y1)
    return float(default_value)


def _evaluate_schedule(
    *,
    mode: str,
    wind_f_mps: float,
    beta_f_deg: float,
    sigmoid_center: float,
    sigmoid_width: float,
    sigmoid_lo: float,
    sigmoid_hi: float,
    rbf_table: list[tuple[float, float, float]],
    rbf_v_sigma: float,
    rbf_beta_sigma_deg: float,
    default_value: float,
    log_interp: bool,
) -> float:
    mode = str(mode)
    if mode in ("", "none"):
        return float(default_value)
    if mode == "sigmoid_v":
        width = max(abs(float(sigmoid_width)), 1e-9)
        s = _sigmoid((float(wind_f_mps) - float(sigmoid_center)) / width)
        return float(sigmoid_lo + (sigmoid_hi - sigmoid_lo) * s)
    if mode == "sigmoid_beta":
        width = max(abs(float(sigmoid_width)), 1e-9)
        s = _sigmoid((float(beta_f_deg) - float(sigmoid_center)) / width)
        return float(sigmoid_lo + (sigmoid_hi - sigmoid_lo) * s)
    if mode == "rbf_vbeta":
        if not rbf_table:
            return float(default_value)
        sig_v = max(abs(float(rbf_v_sigma)), 1e-9)
        sig_b = max(abs(float(rbf_beta_sigma_deg)), 1e-9)
        vals = np.asarray([p[2] for p in rbf_table], dtype=float)
        wts = []
        for v_i, beta_i, _ in rbf_table:
            z_v = (float(wind_f_mps) - float(v_i)) / sig_v
            z_b = (float(beta_f_deg) - float(beta_i)) / sig_b
            wts.append(math.exp(-0.5 * (z_v * z_v + z_b * z_b)))
        w = np.asarray(wts, dtype=float)
        w_sum = float(np.sum(w))
        if (not math.isfinite(w_sum)) or w_sum <= 1e-18:
            return float(default_value)
        if bool(log_interp) and np.all(vals > 0.0):
            return float(math.exp(float(np.dot(w, np.log(vals)) / w_sum)))
        return float(np.dot(w, vals) / w_sum)
    if mode == "piecewise_beta":
        pts = [(float(beta_i), float(val_i)) for _, beta_i, val_i in rbf_table]
        return _piecewise_interp(
            x=float(beta_f_deg),
            points=pts,
            log_interp=bool(log_interp),
            default_value=float(default_value),
        )
    if mode == "piecewise_v":
        pts = [(float(v_i), float(val_i)) for v_i, _, val_i in rbf_table]
        return _piecewise_interp(
            x=float(wind_f_mps),
            points=pts,
            log_interp=bool(log_interp),
            default_value=float(default_value),
        )
    raise ValueError(f"unknown schedule mode={mode!r}")


def _normalize_linear_constraint_rows(
    A_rows: np.ndarray,
    rhs_upper: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_norms = np.linalg.norm(A_rows, axis=1)
    denom = np.maximum(row_norms, float(eps))
    scales = 1.0 / denom
    return scales[:, None] * A_rows, scales * rhs_upper, row_norms


def _build_Af_B(
    thetas: list[np.ndarray], *, n_f: int, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ragged theta_p to:
      a = Af @ f
      y = a + B @ u
    with B lower-triangular in time.
    """
    Af = np.zeros((N, n_f), dtype=float)
    B = np.zeros((N, N), dtype=float)
    for p in range(1, N + 1):
        th = thetas[p - 1]
        if th.shape[0] != n_f + p:
            raise ValueError(f"theta_p{p:02d} has len {th.shape[0]}, expected {n_f+p}")
        Af[p - 1, :] = th[:n_f]
        B[p - 1, :p] = th[n_f:]
    return Af, B


def _u_blocks_for_mode(u_mode: str) -> int:
    if u_mode == "fixed":
        return 1
    if u_mode == "tv_wind2":
        return 2
    if u_mode == "tv_wind2_centered":
        return 2
    if u_mode == "tv_omega_beta_wind2":
        return 4
    if u_mode == "tv_omega_beta_wind2_centered":
        return 4
    raise ValueError(f"unknown u_mode: {u_mode}")


def _build_Af_B_blocks(
    thetas: list[np.ndarray], *, n_f: int, N: int, u_mode: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Convert ragged theta_p to:
      a = Af @ f
      y = a + (sum_j w_j * B_j) @ u

    Where B_j are lower-triangular matrices capturing scheduled input sensitivity.
    """
    u_blocks = _u_blocks_for_mode(str(u_mode))
    Af = np.zeros((N, n_f), dtype=float)
    Bs = [np.zeros((N, N), dtype=float) for _ in range(u_blocks)]
    for p in range(1, N + 1):
        th = thetas[p - 1]
        exp_len = n_f + u_blocks * p
        if th.shape[0] != exp_len:
            raise ValueError(f"theta_p{p:02d} has len {th.shape[0]}, expected {exp_len} (u_mode={u_mode})")
        Af[p - 1, :] = th[:n_f]
        u_th = th[n_f:]
        for j in range(u_blocks):
            Bs[j][p - 1, :p] = u_th[j * p : (j + 1) * p]
    return Af, Bs


def _effective_B(
    *,
    blocks: list[np.ndarray],
    u_mode: str,
    sched_ref: Optional[dict],
    meas: Dict[str, float],
    wind_mps: int,
) -> np.ndarray:
    if str(u_mode) == "fixed":
        return blocks[0]
    if str(u_mode) == "tv_wind2":
        V = _safe_float(meas.get("HorWindV", wind_mps), float(wind_mps))
        return blocks[0] + (V * V) * blocks[1]
    if str(u_mode) == "tv_wind2_centered":
        if sched_ref is None:
            raise ValueError("tv_wind2_centered requires u_sched_ref in model meta")
        V = _safe_float(meas.get("HorWindV", wind_mps), float(wind_mps))
        wind_ref = _safe_float(sched_ref.get("wind_ref", wind_mps), float(wind_mps))
        v2_ref = max(wind_ref * wind_ref, 1e-12)
        dv2 = ((V * V) - v2_ref) / v2_ref
        return blocks[0] + dv2 * blocks[1]
    if str(u_mode) == "tv_omega_beta_wind2":
        V = _safe_float(meas.get("HorWindV", wind_mps), float(wind_mps))
        omega = _safe_float(meas.get("RotSpeed", 0.0), 0.0)
        beta = _safe_float(meas.get("BlPitchCMeas", 0.0), 0.0)
        v2 = V * V
        return blocks[0] + omega * blocks[1] + beta * blocks[2] + v2 * blocks[3]
    if str(u_mode) == "tv_omega_beta_wind2_centered":
        if sched_ref is None:
            raise ValueError("tv_omega_beta_wind2_centered requires u_sched_ref in model meta")
        V = _safe_float(meas.get("HorWindV", wind_mps), float(wind_mps))
        omega = _safe_float(meas.get("RotSpeed", 0.0), 0.0)
        beta = _safe_float(meas.get("BlPitchCMeas", 0.0), 0.0)
        omega_ref = _safe_float(sched_ref.get("omega_ref", omega if abs(omega) > 1e-9 else 1.0), 1.0)
        beta_ref = _safe_float(sched_ref.get("beta_ref", beta if abs(beta) > 1e-9 else 1.0), 1.0)
        wind_ref = _safe_float(sched_ref.get("wind_ref", wind_mps), float(wind_mps))
        v2_ref = max(wind_ref * wind_ref, 1e-12)
        domega = (omega - omega_ref) / max(abs(omega_ref), 1e-12)
        dbeta = (beta - beta_ref) / max(abs(beta_ref), 1e-12)
        dv2 = ((V * V) - v2_ref) / v2_ref
        return blocks[0] + domega * blocks[1] + dbeta * blocks[2] + dv2 * blocks[3]
    raise ValueError(f"unsupported u_mode={u_mode!r}")


def _fir_to_toeplitz(h: np.ndarray, N: int) -> np.ndarray:
    col = np.zeros(N, dtype=float)
    col[: min(N, len(h))] = h[: min(N, len(h))]
    row = np.zeros(N, dtype=float)
    row[0] = float(h[0]) if len(h) else 0.0
    return toeplitz(col, row)


def _bandpass_energy_matrix(
    *,
    N: int,
    ts: float,
    center_hz: float,
    halfwidth_hz: float,
    taps: int,
) -> np.ndarray:
    """
    Build M = H^T H for a band-pass FIR filter H (Toeplitz conv matrix).

    This is PSD and safe to use in a convex quadratic objective:
        (a + B u)^T M (a + B u).
    """
    center_hz = float(center_hz)
    if not math.isfinite(center_hz) or center_hz <= 0:
        return np.zeros((N, N), dtype=float)
    fs = 1.0 / float(ts)
    f_lo = max(1e-6, center_hz - float(halfwidth_hz))
    f_hi = center_hz + float(halfwidth_hz)
    if f_hi >= (0.5 * fs):
        return np.zeros((N, N), dtype=float)
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    if taps > N:
        taps = N if (N % 2 == 1) else (N - 1)
    h = firwin(taps, [f_lo, f_hi], pass_zero=False, fs=fs)
    H = _fir_to_toeplitz(h, N)
    return (H.T @ H).astype(float)


def _first_diff_energy_matrix(N: int, ts: float) -> np.ndarray:
    D1 = np.zeros((N, N), dtype=float)
    scale = 1.0 / math.sqrt(max(float(ts), 1e-12))
    for i in range(N - 1):
        D1[i, i] = -scale
        D1[i, i + 1] = scale
    return D1


def _second_diff_energy_matrix(N: int, ts: float) -> np.ndarray:
    D2 = np.zeros((N, N), dtype=float)
    scale = 1.0 / max(float(ts), 1e-12) ** 1.5
    for i in range(N - 2):
        D2[i, i] = scale
        D2[i, i + 1] = -2.0 * scale
        D2[i, i + 2] = scale
    return D2


def _normalize_metric_matrix(M_ref: np.ndarray, M_new: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode)
    if mode == "none":
        return np.asarray(M_new, dtype=float)
    eps = 1e-18
    if mode == "trace":
        num = float(np.trace(M_ref))
        den = float(np.trace(M_new))
    elif mode == "mean_diag":
        num = float(np.mean(np.diag(M_ref)))
        den = float(np.mean(np.diag(M_new)))
    elif mode == "max_diag":
        num = float(np.max(np.diag(M_ref)))
        den = float(np.max(np.diag(M_new)))
    else:
        raise ValueError(f"unknown tower metric normalize mode {mode!r}")
    if not math.isfinite(den) or abs(den) <= eps:
        return np.asarray(M_ref, dtype=float)
    return (float(num) / float(den)) * np.asarray(M_new, dtype=float)


def _build_pafs_metric(
    *,
    H_tower: np.ndarray,
    a_load: np.ndarray,
    alpha: float,
    m_w: float,
    eps: float,
    normalize_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    z0 = np.asarray(H_tower @ np.asarray(a_load, dtype=float), dtype=float).reshape(-1)
    w, stats = _build_pafs_weights(
        z0=z0,
        alpha=float(alpha),
        m_w=float(m_w),
        eps=float(eps),
    )
    W = np.diag(w.astype(float))
    M_ref = H_tower.T @ H_tower
    M_new = H_tower.T @ W @ H_tower
    M_new = _normalize_metric_matrix(M_ref=M_ref, M_new=M_new, mode=str(normalize_mode))
    return np.asarray(M_new, dtype=float), stats


def _build_pafs_weights(
    *,
    z0: np.ndarray,
    alpha: float,
    m_w: float,
    eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    m_w = float(max(m_w, 2.0))
    eps = float(max(eps, 1e-12))
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    z_abs = np.abs(z0)
    z_inf = float(max(np.max(z_abs), eps)) if z_abs.size else float(eps)
    exponent = float(m_w - 2.0)
    if exponent <= 0.0:
        w = np.ones_like(z_abs, dtype=float)
    else:
        w = alpha + (1.0 - alpha) * np.power(z_abs / z_inf, exponent)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or not math.isfinite(w_sum):
        w_top10_share = float("nan")
    else:
        n_top = max(1, int(math.ceil(0.1 * len(w))))
        w_sorted = np.sort(w)[::-1]
        w_top10_share = float(np.sum(w_sorted[:n_top]) / w_sum)
    stats = {
        "pafs_z0_rms": float(math.sqrt(np.mean(z0 * z0))) if z0.size else float("nan"),
        "pafs_z0_inf": float(z_inf),
        "pafs_w_mean": float(np.mean(w)) if w.size else float("nan"),
        "pafs_w_p95": float(np.quantile(w, 0.95)) if w.size else float("nan"),
        "pafs_w_top10_share": float(w_top10_share),
    }
    return w.astype(float), stats


def _build_hybrid_fatigue_metric(
    *,
    H_tower: np.ndarray,
    a_load: np.ndarray,
    ts: float,
    alpha0: float,
    alpha2: float,
    alpha4: float,
    alphae: float,
    pafs_alpha: float,
    pafs_mw: float,
    pafs_eps: float,
    normalize_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    z0 = np.asarray(H_tower @ np.asarray(a_load, dtype=float), dtype=float).reshape(-1)
    w_env, stats = _build_pafs_weights(
        z0=z0,
        alpha=float(pafs_alpha),
        m_w=float(pafs_mw),
        eps=float(pafs_eps),
    )
    M_ref = H_tower.T @ H_tower
    W_env = np.diag(w_env)
    D1 = _first_diff_energy_matrix(N=int(H_tower.shape[0]), ts=float(ts))
    D2 = _second_diff_energy_matrix(N=int(H_tower.shape[0]), ts=float(ts))
    M0 = M_ref
    M2 = H_tower.T @ D1.T @ D1 @ H_tower
    M4 = H_tower.T @ D2.T @ D2 @ H_tower
    M_env = H_tower.T @ W_env @ H_tower
    M_raw = (
        float(alpha0) * M0
        + float(alpha2) * M2
        + float(alpha4) * M4
        + float(alphae) * M_env
    )
    M_new = _normalize_metric_matrix(M_ref=M_ref, M_new=M_raw, mode=str(normalize_mode))
    trace_raw = float(np.trace(M_raw))
    env_trace = float(np.trace(float(alphae) * M_env))
    spec_trace = float(np.trace(float(alpha0) * M0 + float(alpha2) * M2 + float(alpha4) * M4))
    stats.update(
        {
            "hybrid_env_trace_frac_raw": float(env_trace / trace_raw) if trace_raw > 0.0 else float("nan"),
            "hybrid_spec_trace_frac_raw": float(spec_trace / trace_raw) if trace_raw > 0.0 else float("nan"),
        }
    )
    return np.asarray(M_new, dtype=float), stats


def _build_env_quantile_weights(
    *,
    z0: np.ndarray,
    quantile: float,
    alpha_floor: float,
    sharpness: float,
    eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    z_abs = np.abs(z0)
    if z_abs.size == 0:
        w = np.zeros((0,), dtype=float)
        return w, {"tau": float("nan"), "scale": float("nan"), "w_mean": float("nan"), "w_p95": float("nan")}
    quantile = float(np.clip(quantile, 0.0, 1.0))
    alpha_floor = float(np.clip(alpha_floor, 0.0, 1.0))
    sharpness = float(max(sharpness, 1e-6))
    eps = float(max(eps, 1e-12))
    tau = float(np.quantile(z_abs, quantile))
    q_hi = float(np.quantile(z_abs, min(0.98, max(quantile + 0.1, quantile))))
    scale = max((q_hi - tau) / sharpness, eps)
    arg = np.clip((z_abs - tau) / scale, -60.0, 60.0)
    gate = 1.0 / (1.0 + np.exp(-arg))
    w = alpha_floor + (1.0 - alpha_floor) * gate
    return w.astype(float), {
        "tau": tau,
        "scale": float(scale),
        "w_mean": float(np.mean(w)),
        "w_p95": float(np.quantile(w, 0.95)),
    }


def _lowpass_series(x: np.ndarray, *, ts: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    fs = 1.0 / float(ts)
    wn = float(cutoff_hz) / (0.5 * fs)
    if (not math.isfinite(wn)) or wn <= 0.0:
        return x.copy()
    wn = min(0.99, wn)
    b, a = butter(int(order), wn, btype="low")
    padlen = min(int(3 * max(len(a), len(b))), max(0, x.size - 1))
    if x.size < max(8, int(3 * max(len(a), len(b)))):
        return x.copy()
    return np.asarray(filtfilt(b, a, x, padlen=padlen), dtype=float)


def _build_turning_point_weights(
    *,
    env_lp: np.ndarray,
    quantile: float,
    alpha_floor: float,
    window_steps: int,
    eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    env_lp = np.asarray(env_lp, dtype=float).reshape(-1)
    n = int(env_lp.size)
    if n == 0:
        w = np.zeros((0,), dtype=float)
        return w, {
            "tau": float("nan"),
            "n_peaks": 0.0,
            "w_mean": float("nan"),
            "w_p95": float("nan"),
        }
    alpha_floor = float(np.clip(alpha_floor, 0.0, 1.0))
    eps = float(max(eps, 1e-12))
    tau = float(np.quantile(env_lp, float(np.clip(quantile, 0.0, 1.0))))
    peaks = np.zeros((n,), dtype=float)
    for i in range(1, n - 1):
        if env_lp[i] >= env_lp[i - 1] and env_lp[i] >= env_lp[i + 1] and env_lp[i] >= tau:
            peaks[i] = float(max(0.0, env_lp[i] - tau))
    if float(np.max(peaks)) > eps:
        peaks = peaks / float(np.max(peaks))
    radius = int(max(1, window_steps))
    kernel = np.arange(1, radius + 2, dtype=float)
    kernel = np.concatenate([kernel, kernel[-2::-1]])
    kernel = kernel / float(np.max(kernel))
    gate = np.convolve(peaks, kernel, mode="same")
    if gate.size != n:
        start = max(0, (gate.size - n) // 2)
        gate = gate[start : start + n]
        if gate.size != n:
            gate = np.resize(gate, n)
    if float(np.max(gate)) > eps:
        gate = gate / float(np.max(gate))
    gate = np.clip(gate, 0.0, 1.0)
    w = alpha_floor + (1.0 - alpha_floor) * gate
    return w.astype(float), {
        "tau": float(tau),
        "n_peaks": float(np.count_nonzero(peaks > 0.0)),
        "w_mean": float(np.mean(w)),
        "w_p95": float(np.quantile(w, 0.95)),
    }


def _sideband_pair_energy_matrix(
    *,
    N: int,
    ts: float,
    carrier_hz: float,
    offset_hz: float,
    halfwidth_hz: float,
    taps: int,
) -> np.ndarray:
    carrier_hz = float(carrier_hz)
    offset_hz = float(offset_hz)
    if (not math.isfinite(carrier_hz)) or (not math.isfinite(offset_hz)) or offset_hz <= 0.0:
        return np.zeros((N, N), dtype=float)
    mats = []
    for center in (carrier_hz - offset_hz, carrier_hz + offset_hz):
        if center <= 0.0:
            continue
        M_c = _bandpass_energy_matrix(
            N=N,
            ts=float(ts),
            center_hz=float(center),
            halfwidth_hz=float(halfwidth_hz),
            taps=int(taps),
        )
        if np.any(M_c):
            mats.append(M_c)
    if not mats:
        return np.zeros((N, N), dtype=float)
    return np.sum(np.stack(mats, axis=0), axis=0)


def _build_psd_cone_metric(
    *,
    H_tower: np.ndarray,
    a_load: np.ndarray,
    ts: float,
    bp_center_hz: float,
    bp_halfwidth_hz: float,
    bp_taps: int,
    f_1p_ref_hz: float,
    alpha_band: float,
    alpha_m2: float,
    alpha_m4: float,
    alpha_env70: float,
    alpha_env85: float,
    alpha_env95: float,
    alpha_envlp85: float,
    alpha_envlp95: float,
    alpha_tp90: float,
    alpha_tp95: float,
    alpha_sb1p: float,
    alpha_sb3p: float,
    alpha_sb1p_env95: float,
    alpha_sb3p_env95: float,
    env_floor: float,
    env_sharpness: float,
    lowfreq_cutoff_hz: float,
    tp_window_s: float,
    basis_normalize: str,
    normalize_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    _, _, M_full, stats = _build_psd_cone_components(
        H_tower=H_tower,
        a_load=a_load,
        ts=ts,
        bp_center_hz=bp_center_hz,
        bp_halfwidth_hz=bp_halfwidth_hz,
        bp_taps=bp_taps,
        f_1p_ref_hz=f_1p_ref_hz,
        alpha_band=alpha_band,
        alpha_m2=alpha_m2,
        alpha_m4=alpha_m4,
        alpha_env70=alpha_env70,
        alpha_env85=alpha_env85,
        alpha_env95=alpha_env95,
        alpha_envlp85=alpha_envlp85,
        alpha_envlp95=alpha_envlp95,
        alpha_tp90=alpha_tp90,
        alpha_tp95=alpha_tp95,
        alpha_sb1p=alpha_sb1p,
        alpha_sb3p=alpha_sb3p,
        alpha_sb1p_env95=alpha_sb1p_env95,
        alpha_sb3p_env95=alpha_sb3p_env95,
        env_floor=env_floor,
        env_sharpness=env_sharpness,
        lowfreq_cutoff_hz=lowfreq_cutoff_hz,
        tp_window_s=tp_window_s,
        basis_normalize=basis_normalize,
        normalize_mode=normalize_mode,
    )
    return np.asarray(M_full, dtype=float), stats


def _build_psd_cone_components(
    *,
    H_tower: np.ndarray,
    a_load: np.ndarray,
    ts: float,
    bp_center_hz: float,
    bp_halfwidth_hz: float,
    bp_taps: int,
    f_1p_ref_hz: float,
    alpha_band: float,
    alpha_m2: float,
    alpha_m4: float,
    alpha_env70: float,
    alpha_env85: float,
    alpha_env95: float,
    alpha_envlp85: float,
    alpha_envlp95: float,
    alpha_tp90: float,
    alpha_tp95: float,
    alpha_sb1p: float,
    alpha_sb3p: float,
    alpha_sb1p_env95: float,
    alpha_sb3p_env95: float,
    env_floor: float,
    env_sharpness: float,
    lowfreq_cutoff_hz: float,
    tp_window_s: float,
    basis_normalize: str,
    normalize_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    N = int(H_tower.shape[0])
    M_ref = H_tower.T @ H_tower
    z0 = np.asarray(H_tower @ np.asarray(a_load, dtype=float), dtype=float).reshape(-1)
    D1 = _first_diff_energy_matrix(N=N, ts=float(ts))
    D2 = _second_diff_energy_matrix(N=N, ts=float(ts))
    M2_raw = H_tower.T @ D1.T @ D1 @ H_tower
    M4_raw = H_tower.T @ D2.T @ D2 @ H_tower
    w70, st70 = _build_env_quantile_weights(z0=z0, quantile=0.70, alpha_floor=float(env_floor), sharpness=float(env_sharpness), eps=1e-8)
    w85, st85 = _build_env_quantile_weights(z0=z0, quantile=0.85, alpha_floor=float(env_floor), sharpness=float(env_sharpness), eps=1e-8)
    w95, st95 = _build_env_quantile_weights(z0=z0, quantile=0.95, alpha_floor=float(env_floor), sharpness=float(env_sharpness), eps=1e-8)
    env_lp = _lowpass_series(np.abs(hilbert(z0)), ts=float(ts), cutoff_hz=float(lowfreq_cutoff_hz), order=2)
    w_envlp85, stlp85 = _build_env_quantile_weights(z0=env_lp, quantile=0.85, alpha_floor=float(env_floor), sharpness=float(env_sharpness), eps=1e-8)
    w_envlp95, stlp95 = _build_env_quantile_weights(z0=env_lp, quantile=0.95, alpha_floor=float(env_floor), sharpness=float(env_sharpness), eps=1e-8)
    tp_window_steps = int(max(1, round(float(tp_window_s) / max(float(ts), 1e-12))))
    w_tp90, sttp90 = _build_turning_point_weights(
        env_lp=env_lp,
        quantile=0.90,
        alpha_floor=float(env_floor),
        window_steps=tp_window_steps,
        eps=1e-8,
    )
    w_tp95, sttp95 = _build_turning_point_weights(
        env_lp=env_lp,
        quantile=0.95,
        alpha_floor=float(env_floor),
        window_steps=tp_window_steps,
        eps=1e-8,
    )
    M_env70_raw = H_tower.T @ np.diag(w70) @ H_tower
    M_env85_raw = H_tower.T @ np.diag(w85) @ H_tower
    M_env95_raw = H_tower.T @ np.diag(w95) @ H_tower
    M_envlp85_raw = H_tower.T @ np.diag(w_envlp85) @ H_tower
    M_envlp95_raw = H_tower.T @ np.diag(w_envlp95) @ H_tower
    M_tp90_raw = H_tower.T @ np.diag(w_tp90) @ H_tower
    M_tp95_raw = H_tower.T @ np.diag(w_tp95) @ H_tower
    M_sb1p_raw = _sideband_pair_energy_matrix(
        N=N,
        ts=float(ts),
        carrier_hz=float(bp_center_hz),
        offset_hz=float(f_1p_ref_hz),
        halfwidth_hz=float(bp_halfwidth_hz),
        taps=int(bp_taps),
    )
    M_sb3p_raw = _sideband_pair_energy_matrix(
        N=N,
        ts=float(ts),
        carrier_hz=float(bp_center_hz),
        offset_hz=3.0 * float(f_1p_ref_hz),
        halfwidth_hz=float(bp_halfwidth_hz),
        taps=int(bp_taps),
    )
    E95 = np.diag(np.sqrt(np.asarray(w95, dtype=float)))
    M_sb1p_env95_raw = E95 @ M_sb1p_raw @ E95 if np.any(M_sb1p_raw) else np.zeros_like(M_ref)
    M_sb3p_env95_raw = E95 @ M_sb3p_raw @ E95 if np.any(M_sb3p_raw) else np.zeros_like(M_ref)

    def _basis(M_raw: np.ndarray) -> np.ndarray:
        return _normalize_metric_matrix(M_ref=M_ref, M_new=M_raw, mode=str(basis_normalize))

    pieces = {
        "band": (float(alpha_band), M_ref),
        "m2": (float(alpha_m2), _basis(M2_raw)),
        "m4": (float(alpha_m4), _basis(M4_raw)),
        "env70": (float(alpha_env70), _basis(M_env70_raw)),
        "env85": (float(alpha_env85), _basis(M_env85_raw)),
        "env95": (float(alpha_env95), _basis(M_env95_raw)),
        "envlp85": (float(alpha_envlp85), _basis(M_envlp85_raw)),
        "envlp95": (float(alpha_envlp95), _basis(M_envlp95_raw)),
        "tp90": (float(alpha_tp90), _basis(M_tp90_raw)),
        "tp95": (float(alpha_tp95), _basis(M_tp95_raw)),
        "sb1p": (float(alpha_sb1p), _basis(M_sb1p_raw)),
        "sb3p": (float(alpha_sb3p), _basis(M_sb3p_raw)),
        "sb1p_env95": (float(alpha_sb1p_env95), _basis(M_sb1p_env95_raw)),
        "sb3p_env95": (float(alpha_sb3p_env95), _basis(M_sb3p_env95_raw)),
    }
    M_full_raw = np.zeros_like(M_ref, dtype=float)
    M_shape_raw = np.zeros_like(M_ref, dtype=float)
    trace_raw = 0.0
    trace_shape_raw = 0.0
    trace_parts: dict[str, float] = {}
    for name, (alpha, M_part) in pieces.items():
        if alpha <= 0.0:
            trace_parts[name] = 0.0
            continue
        contrib = alpha * np.asarray(M_part, dtype=float)
        M_full_raw += contrib
        tr = float(np.trace(contrib))
        trace_parts[name] = tr
        trace_raw += tr
        if name != "band":
            M_shape_raw += contrib
            trace_shape_raw += tr
    if trace_raw <= 0.0:
        M_full_raw = M_ref.copy()
        trace_raw = float(np.trace(M_full_raw))
        trace_parts["band"] = trace_raw
    M_full = _normalize_metric_matrix(M_ref=M_ref, M_new=M_full_raw, mode=str(normalize_mode))
    if trace_shape_raw > 0.0:
        M_shape = _normalize_metric_matrix(M_ref=M_ref, M_new=M_shape_raw, mode=str(normalize_mode))
    else:
        M_shape = np.zeros_like(M_ref, dtype=float)
    z0_rms = float(np.sqrt(np.mean(z0 * z0))) if z0.size else float("nan")
    env_lp_rms = float(np.sqrt(np.mean(env_lp * env_lp))) if env_lp.size else float("nan")
    envlp95_tau = float(stlp95["tau"])
    env_severity = float(envlp95_tau / max(z0_rms, 1e-12)) if math.isfinite(envlp95_tau) and math.isfinite(z0_rms) else float("nan")
    stats = {
        "pafs_z0_rms": float("nan"),
        "pafs_z0_inf": float("nan"),
        "pafs_w_mean": float("nan"),
        "pafs_w_p95": float("nan"),
        "pafs_w_top10_share": float("nan"),
        "hybrid_env_trace_frac_raw": float("nan"),
        "hybrid_spec_trace_frac_raw": float("nan"),
        "psd_band_trace_frac_raw": float(trace_parts.get("band", 0.0) / trace_raw),
        "psd_m2_trace_frac_raw": float(trace_parts.get("m2", 0.0) / trace_raw),
        "psd_m4_trace_frac_raw": float(trace_parts.get("m4", 0.0) / trace_raw),
        "psd_env70_trace_frac_raw": float(trace_parts.get("env70", 0.0) / trace_raw),
        "psd_env85_trace_frac_raw": float(trace_parts.get("env85", 0.0) / trace_raw),
        "psd_env95_trace_frac_raw": float(trace_parts.get("env95", 0.0) / trace_raw),
        "psd_envlp85_trace_frac_raw": float(trace_parts.get("envlp85", 0.0) / trace_raw),
        "psd_envlp95_trace_frac_raw": float(trace_parts.get("envlp95", 0.0) / trace_raw),
        "psd_tp90_trace_frac_raw": float(trace_parts.get("tp90", 0.0) / trace_raw),
        "psd_tp95_trace_frac_raw": float(trace_parts.get("tp95", 0.0) / trace_raw),
        "psd_sb1p_trace_frac_raw": float(trace_parts.get("sb1p", 0.0) / trace_raw),
        "psd_sb3p_trace_frac_raw": float(trace_parts.get("sb3p", 0.0) / trace_raw),
        "psd_sb1p_env95_trace_frac_raw": float(trace_parts.get("sb1p_env95", 0.0) / trace_raw),
        "psd_sb3p_env95_trace_frac_raw": float(trace_parts.get("sb3p_env95", 0.0) / trace_raw),
        "psd_env_trace_frac_raw": float((trace_parts.get("env70", 0.0) + trace_parts.get("env85", 0.0) + trace_parts.get("env95", 0.0) + trace_parts.get("envlp85", 0.0) + trace_parts.get("envlp95", 0.0) + trace_parts.get("tp90", 0.0) + trace_parts.get("tp95", 0.0)) / trace_raw),
        "psd_spec_trace_frac_raw": float((trace_parts.get("band", 0.0) + trace_parts.get("m2", 0.0) + trace_parts.get("m4", 0.0)) / trace_raw),
        "psd_sideband_trace_frac_raw": float((trace_parts.get("sb1p", 0.0) + trace_parts.get("sb3p", 0.0) + trace_parts.get("sb1p_env95", 0.0) + trace_parts.get("sb3p_env95", 0.0)) / trace_raw),
        "psd_env70_tau": float(st70["tau"]),
        "psd_env85_tau": float(st85["tau"]),
        "psd_env95_tau": float(st95["tau"]),
        "psd_envlp85_tau": float(stlp85["tau"]),
        "psd_envlp95_tau": float(stlp95["tau"]),
        "psd_tp90_tau": float(sttp90["tau"]),
        "psd_tp95_tau": float(sttp95["tau"]),
        "psd_tp90_n_peaks": float(sttp90["n_peaks"]),
        "psd_tp95_n_peaks": float(sttp95["n_peaks"]),
        "psd_z0_rms": float(z0_rms),
        "psd_envlp_rms": float(env_lp_rms),
        "psd_envlp95_tau_abs": float(envlp95_tau),
        "psd_env_severity": float(env_severity),
        "psd_shape_trace_frac_raw": float(trace_shape_raw / trace_raw) if trace_raw > 0.0 else float("nan"),
    }
    return np.asarray(M_ref, dtype=float), np.asarray(M_shape, dtype=float), np.asarray(M_full, dtype=float), stats


def _structured_gate_value(
    *,
    severity: float,
    env_center: float,
    env_width: float,
    op_mode: str,
    wind_f_mps: float,
    beta_f_deg: float,
    op_center: float,
    op_width: float,
    op_lo: float,
    op_hi: float,
    op_rbf_table: list[tuple[float, float, float]],
    op_rbf_v_sigma: float,
    op_rbf_beta_sigma_deg: float,
) -> tuple[float, float, float]:
    width = max(abs(float(env_width)), 1e-9)
    env_gate = _sigmoid((float(severity) - float(env_center)) / width)
    op_gate = _evaluate_schedule(
        mode=str(op_mode),
        wind_f_mps=float(wind_f_mps),
        beta_f_deg=float(beta_f_deg),
        sigmoid_center=float(op_center),
        sigmoid_width=float(op_width),
        sigmoid_lo=float(op_lo),
        sigmoid_hi=float(op_hi),
        rbf_table=op_rbf_table,
        rbf_v_sigma=float(op_rbf_v_sigma),
        rbf_beta_sigma_deg=float(op_rbf_beta_sigma_deg),
        default_value=1.0,
        log_interp=False,
    )
    op_gate = float(np.clip(op_gate, 0.0, 1.0))
    return float(env_gate * op_gate), float(env_gate), float(op_gate)


def _quadratic_prediction_cost(a: np.ndarray, B: np.ndarray, M: np.ndarray, u: np.ndarray) -> float:
    y = np.asarray(a, dtype=float) + np.asarray(B, dtype=float) @ np.asarray(u, dtype=float)
    return float(y.T @ np.asarray(M, dtype=float) @ y)


def _bandpass_toeplitz(
    *,
    N: int,
    ts: float,
    center_hz: float,
    halfwidth_hz: float,
    taps: int,
) -> np.ndarray:
    center_hz = float(center_hz)
    if not math.isfinite(center_hz) or center_hz <= 0:
        return np.zeros((N, N), dtype=float)
    fs = 1.0 / float(ts)
    f_lo = max(1e-6, center_hz - float(halfwidth_hz))
    f_hi = center_hz + float(halfwidth_hz)
    if f_hi >= (0.5 * fs):
        return np.zeros((N, N), dtype=float)
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    if taps > N:
        taps = N if (N % 2 == 1) else (N - 1)
    if taps < 3:
        taps = 3 if N >= 3 else N
    h = firwin(taps, [f_lo, f_hi], pass_zero=False, fs=fs)
    return _fir_to_toeplitz(h, N).astype(float)


def _quantize_freq_hz(f_hz: float, step_hz: float) -> float:
    f_hz = float(f_hz)
    step_hz = float(step_hz)
    if (not math.isfinite(f_hz)) or (not math.isfinite(step_hz)) or step_hz <= 0:
        return f_hz
    return step_hz * float(np.round(f_hz / step_hz))


def _load_band_quant_step_hz(qp: "QPBundle") -> float:
    steps = []
    if float(qp.load_rel_1p) > 0.0:
        steps.append(float(qp.load_1p_halfwidth_hz))
    if float(qp.load_rel_3p) > 0.0:
        steps.append(float(qp.load_3p_halfwidth_hz) / 3.0)
    if not steps:
        return 0.0
    return float(max(0.002, 0.25 * min(steps)))


def _get_M_total_and_maybe_update_cache(
    qp: "QPBundle",
    *,
    rot_speed_radps: float,
) -> tuple[np.ndarray, bool]:
    """
    Return M_total used in the load objective, and whether it changed vs cache.

    If load_freq_mode == 'meas', 1P/3P band centers follow the measured rotor speed
    (rad/s -> Hz), quantized to a small step to avoid excessive recomputation.
    """
    if float(qp.load_rel_1p) <= 0.0 and float(qp.load_rel_3p) <= 0.0:
        return qp.M_tower, False

    mode = str(qp.load_freq_mode)
    if mode not in ("ref", "meas"):
        raise ValueError(f"unknown load_freq_mode={mode!r}")

    f_1p_hz = float(qp.f_1p_ref_hz)
    if mode == "meas":
        omega = float(rot_speed_radps)
        if math.isfinite(omega) and omega > 0:
            f_1p_hz = omega / (2.0 * math.pi)

    if not (math.isfinite(f_1p_hz) and f_1p_hz > 0):
        # Safe fallback: only tower band.
        return qp.M_tower, False

    f_q = _quantize_freq_hz(f_1p_hz, _load_band_quant_step_hz(qp))
    if qp.M_cache is not None and math.isfinite(qp.f_1p_cache_hz):
        if abs(float(qp.f_1p_cache_hz) - float(f_q)) <= 1e-12:
            return qp.M_cache, False

    M_total = qp.M_tower
    if float(qp.load_rel_1p) > 0.0:
        M_1p = _bandpass_energy_matrix(
            N=qp.M_tower.shape[0],
            ts=float(qp.ts),
            center_hz=float(f_q),
            halfwidth_hz=float(qp.load_1p_halfwidth_hz),
            taps=int(qp.load_1p_taps),
        )
        M_total = M_total + float(qp.load_rel_1p) * M_1p
    if float(qp.load_rel_3p) > 0.0:
        M_3p = _bandpass_energy_matrix(
            N=qp.M_tower.shape[0],
            ts=float(qp.ts),
            center_hz=float(3.0 * f_q),
            halfwidth_hz=float(qp.load_3p_halfwidth_hz),
            taps=int(qp.load_3p_taps),
        )
        M_total = M_total + float(qp.load_rel_3p) * M_3p

    qp.f_1p_cache_hz = float(f_q)
    qp.M_cache = M_total
    return M_total, True


def _diff_matrix(N: int) -> np.ndarray:
    D = np.zeros((N, N), dtype=float)
    D[0, 0] = 1.0
    for i in range(1, N):
        D[i, i] = 1.0
        D[i, i - 1] = -1.0
    return D


def _diff_chain_matrix(N: int) -> np.ndarray:
    if int(N) <= 1:
        return np.zeros((0, int(N)), dtype=float)
    D = np.zeros((int(N) - 1, int(N)), dtype=float)
    for i in range(1, int(N)):
        D[i - 1, i] = 1.0
        D[i - 1, i - 1] = -1.0
    return D


def _parse_move_blocking_spec(spec: str, *, N: int) -> list[tuple[int, int]]:
    spec = str(spec).strip()
    if spec in ("", "none", "full", "step"):
        return [(i, i + 1) for i in range(int(N))]
    blocks: list[tuple[int, int]] = []
    cursor = 0
    for raw_tok in spec.replace("*", "x").split(","):
        tok = raw_tok.strip().lower()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"invalid move-block token {raw_tok!r}; expected countxsteps, e.g. 20x1")
        count_s, span_s = tok.split("x", 1)
        count = int(count_s)
        span = int(span_s)
        if count <= 0 or span <= 0:
            raise ValueError(f"invalid move-block token {raw_tok!r}; count/span must be positive")
        for _ in range(count):
            start = int(cursor)
            end = int(cursor + span)
            blocks.append((start, end))
            cursor = end
    if cursor != int(N):
        raise ValueError(f"move-blocking spec {spec!r} covers {cursor} steps, expected {int(N)}")
    return blocks


def _move_blocking_equalities(blocks: list[tuple[int, int]], *, N: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start, end in blocks:
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i <= 1:
            continue
        for idx in range(start_i + 1, end_i):
            row = np.zeros((int(N),), dtype=float)
            row[start_i] = -1.0
            row[int(idx)] = 1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, int(N)), dtype=float)
    return np.vstack(rows)


def _csc_triu_indices(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, indptr) for a CSC matrix storing the full upper triangle
    (including diagonal) of an (N,N) symmetric matrix.

    Column j stores rows 0..j (length j+1). This fixed sparsity pattern is
    compatible with OSQP's requirement that P is provided in upper-triangular form.
    """
    indptr = np.zeros((N + 1,), dtype=np.int32)
    for j in range(N):
        indptr[j + 1] = indptr[j] + (j + 1)
    nnz = int(indptr[-1])
    indices = np.empty((nnz,), dtype=np.int32)
    k = 0
    for j in range(N):
        n = j + 1
        indices[k : k + n] = np.arange(n, dtype=np.int32)
        k += n
    return indices, indptr


def _csc_triu_data(P: np.ndarray) -> np.ndarray:
    """
    Pack the upper triangle (including diagonal) of a dense (N,N) matrix P into
    a CSC data array matching `_csc_triu_indices(N)`.
    """
    N = int(P.shape[0])
    out = np.empty((N * (N + 1) // 2,), dtype=float)
    k = 0
    for j in range(N):
        n = j + 1
        out[k : k + n] = P[:n, j]
        k += n
    return out


def _control_curvature_metric(Q: np.ndarray, *, mode: str) -> float:
    mode = str(mode)
    if mode in ("", "none"):
        return 1.0
    diag = np.diag(Q)
    if mode == "trace":
        val = float(np.trace(Q))
    elif mode == "mean_diag":
        val = float(np.mean(diag))
    elif mode == "max_diag":
        val = float(np.max(diag))
    else:
        raise ValueError(f"unknown g_norm_mode={mode!r}")
    return float(val)


def _normalized_B_for_qp(
    *,
    B_ref: np.ndarray,
    B_eff: np.ndarray,
    M_total: np.ndarray,
    mode: str,
    form: str,
    scale_min: float,
    scale_max: float,
    eps: float,
) -> tuple[np.ndarray, float, float, float]:
    if str(mode) == "none":
        return B_eff, 1.0, 1.0, 1.0
    Q_ref = B_ref.T @ M_total @ B_ref
    Q_eff = B_eff.T @ M_total @ B_eff
    metric_ref = _control_curvature_metric(Q_ref, mode=str(mode))
    metric_eff = _control_curvature_metric(Q_eff, mode=str(mode))
    eps = max(float(eps), 1e-24)
    scale = 1.0
    if math.isfinite(metric_ref) and math.isfinite(metric_eff) and metric_ref > eps and metric_eff > eps:
        scale = math.sqrt(metric_ref / metric_eff)
    scale = float(np.clip(scale, float(scale_min), float(scale_max)))
    form = str(form)
    if form == "scale_B":
        B_qp = float(scale) * B_eff
    elif form == "metric_only":
        B_qp = np.asarray(B_eff, dtype=float)
    else:
        raise ValueError(f"unknown g_norm_form={form!r}")
    return B_qp, float(scale), float(metric_ref), float(metric_eff)


def _svd_energy_lowrank(
    B: np.ndarray,
    *,
    energy_retain: float,
    min_rank: int,
    max_rank: int,
) -> tuple[np.ndarray, int, float]:
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.size == 0:
        return B.copy(), 0, float("nan")
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    if s.size == 0:
        return B.copy(), 0, float("nan")
    energy = s * s
    total = float(np.sum(energy))
    if not math.isfinite(total) or total <= 0.0:
        rank = max(1, min(int(min_rank), s.size))
        if int(max_rank) > 0:
            rank = min(rank, int(max_rank))
        B_lr = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
        return B_lr, int(rank), float("nan")
    retain = float(np.clip(float(energy_retain), 0.0, 1.0))
    cum = np.cumsum(energy) / total
    rank = int(np.searchsorted(cum, retain, side="left")) + 1
    rank = max(int(min_rank), rank)
    if int(max_rank) > 0:
        rank = min(rank, int(max_rank))
    rank = min(rank, s.size)
    B_lr = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    frac = float(np.sum(energy[:rank]) / total)
    return B_lr, int(rank), float(frac)


def _embed_u_matrix(mat_u: np.ndarray, *, n_var: int) -> np.ndarray:
    n_u = int(mat_u.shape[0])
    out = np.zeros((int(n_var), int(n_var)), dtype=float)
    out[:n_u, :n_u] = np.asarray(mat_u, dtype=float)
    return out


def _embed_u_vector(vec_u: np.ndarray, *, n_var: int) -> np.ndarray:
    n_u = int(vec_u.shape[0])
    out = np.zeros(int(n_var), dtype=float)
    out[:n_u] = np.asarray(vec_u, dtype=float)
    return out


def _cumulative_sum_matrix(n_u: int) -> np.ndarray:
    return np.tril(np.ones((int(n_u), int(n_u)), dtype=float))


def _active_rows_from_bounds(A: np.ndarray, x: np.ndarray, l: np.ndarray, u: np.ndarray, *, tol: float = 1e-7) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    l = np.asarray(l, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    Ax = A @ x
    rows: list[np.ndarray] = []
    for i in range(A.shape[0]):
        li = float(l[i])
        ui = float(u[i])
        ai = A[i]
        if math.isfinite(li) and abs(float(Ax[i]) - li) <= float(tol):
            rows.append(ai.copy())
        elif math.isfinite(ui) and abs(float(Ax[i]) - ui) <= float(tol):
            rows.append(ai.copy())
    if not rows:
        return np.zeros((0, A.shape[1]), dtype=float)
    return np.vstack(rows)


def _solve_kkt_sensitivity(P: np.ndarray, A_act: np.ndarray, rhs_obj: np.ndarray) -> tuple[np.ndarray, float, int]:
    P = np.asarray(P, dtype=float)
    rhs_obj = np.asarray(rhs_obj, dtype=float).reshape(-1)
    A_act = np.asarray(A_act, dtype=float)
    if A_act.size == 0:
        try:
            du = -np.linalg.solve(P, rhs_obj)
            return du, float(np.linalg.cond(P)), int(np.linalg.matrix_rank(P))
        except np.linalg.LinAlgError:
            return np.full_like(rhs_obj, np.nan), float("inf"), 0
    m = int(A_act.shape[0])
    n = int(P.shape[0])
    K = np.zeros((n + m, n + m), dtype=float)
    K[:n, :n] = P
    K[:n, n:] = A_act.T
    K[n:, :n] = A_act
    rhs = np.zeros((n + m,), dtype=float)
    rhs[:n] = -rhs_obj
    try:
        sol = np.linalg.solve(K, rhs)
        cond = float(np.linalg.cond(K))
        rank = int(np.linalg.matrix_rank(A_act))
        return np.asarray(sol[:n], dtype=float), cond, rank
    except np.linalg.LinAlgError:
        return np.full((n,), np.nan, dtype=float), float("inf"), 0


def _build_runtime_constraint_system(
    *,
    qp: "QPBundle",
    args: argparse.Namespace,
    meas: Dict[str, float],
    u_prev: float,
    u_max_runtime: float,
    du_max_runtime: float,
    a_pwr: Optional[np.ndarray],
    a_speed: Optional[np.ndarray],
    B_eff_speed: Optional[np.ndarray],
    a_load_primary: Optional[np.ndarray],
    B_qp_primary: Optional[np.ndarray],
    primary_cap_rel: float,
    primary_cap_floor_rel: float,
    a_load_secondary: Optional[np.ndarray],
    B_qp_secondary: Optional[np.ndarray],
    secondary_cap_rel: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict[str, float]]:
    l = qp.base_l.copy()
    u = qp.base_u.copy()
    A_rebuild_dense = None
    speed_guard_rhs_min = float("nan")
    speed_guard_row_norm_mean = float("nan")
    speed_guard_row_norm_max = float("nan")
    tower_primary_cap_floor = float("nan")
    tower_primary_cap_free_rms = float("nan")

    amp_slice = slice(0, int(qp.n_u))
    if str(qp.decision_coords) == "delta_u":
        u_hold = float(u_prev) * qp.ones_u
        l[amp_slice] = -float(u_max_runtime) - u_hold
        u[amp_slice] = +float(u_max_runtime) - u_hold
    else:
        l[amp_slice] = -float(u_max_runtime)
        u[amp_slice] = +float(u_max_runtime)

    rate_slice = slice(int(qp.idx_rate0), int(qp.idx_rate0 + qp.n_u))
    l[rate_slice] = -float(du_max_runtime)
    u[rate_slice] = +float(du_max_runtime)
    if str(qp.decision_coords) != "delta_u":
        l[qp.idx_rate0] = float(u_prev) - float(du_max_runtime)
        u[qp.idx_rate0] = float(u_prev) + float(du_max_runtime)

    if qp.idx_power is not None and qp.power_c is not None and qp.Af_pwr is not None and a_pwr is not None:
        p_meas_kw = float(meas.get("VS_GenPwr", 0.0)) / 1000.0
        rhs = (1.0 - float(args.power_eps)) * p_meas_kw - float(np.mean(a_pwr))
        if str(qp.decision_coords) == "delta_u" and qp.power_c_abs is not None:
            rhs -= float(qp.power_c_abs @ (float(u_prev) * qp.ones_u))
        l[qp.idx_power] = rhs
        u[qp.idx_power] = np.inf

    if (
        qp.idx_speed is not None
        and qp.n_speed_rows > 0
        and a_speed is not None
        and B_eff_speed is not None
    ):
        speed_slice = slice(int(qp.idx_speed), int(qp.idx_speed + qp.n_speed_rows))
        if A_rebuild_dense is None:
            A_rebuild_dense = qp.A_dense_base.copy()
        if str(qp.speed_form) == "step_upper":
            A_speed_rows = B_eff_speed
            rhs_speed = float(qp.speed_upper) - a_speed
            row_norms = np.linalg.norm(A_speed_rows, axis=1)
            if bool(args.speed_row_normalize):
                A_speed_rows, rhs_speed, row_norms = _normalize_linear_constraint_rows(
                    A_speed_rows,
                    rhs_speed,
                    eps=float(args.speed_row_norm_eps),
                )
            A_rebuild_dense[speed_slice, :] = 0.0
            A_rebuild_dense[speed_slice, : qp.n_u] = A_speed_rows
            l[speed_slice] = -np.inf
            u[speed_slice] = rhs_speed
            speed_guard_rhs_min = float(np.min(rhs_speed))
            speed_guard_row_norm_mean = float(np.mean(row_norms))
            speed_guard_row_norm_max = float(np.max(row_norms))
        elif str(qp.speed_form) == "mean_upper":
            A_speed_rows = np.mean(B_eff_speed, axis=0, keepdims=True)
            rhs_speed = np.asarray([float(qp.speed_upper) - float(np.mean(a_speed))], dtype=float)
            row_norms = np.linalg.norm(A_speed_rows, axis=1)
            if bool(args.speed_row_normalize):
                A_speed_rows, rhs_speed, row_norms = _normalize_linear_constraint_rows(
                    A_speed_rows,
                    rhs_speed,
                    eps=float(args.speed_row_norm_eps),
                )
            A_rebuild_dense[speed_slice, :] = 0.0
            A_rebuild_dense[speed_slice, : qp.n_u] = A_speed_rows
            l[speed_slice] = -np.inf
            u[speed_slice] = rhs_speed
            speed_guard_rhs_min = float(rhs_speed[0])
            speed_guard_row_norm_mean = float(row_norms[0])
            speed_guard_row_norm_max = float(row_norms[0])

    if (
        qp.idx_speed_soft is not None
        and qp.n_speed_soft_rows > 0
        and qp.idx_speed_slack is not None
        and a_speed is not None
        and B_eff_speed is not None
    ):
        speed_soft_slice = slice(int(qp.idx_speed_soft), int(qp.idx_speed_soft + qp.n_speed_soft_rows))
        if A_rebuild_dense is None:
            A_rebuild_dense = qp.A_dense_base.copy()
        if str(qp.speed_form) == "step_upper":
            A_speed_soft = B_eff_speed.copy()
            rhs_speed_soft = float(qp.speed_upper) - a_speed
            row_norms_soft = np.linalg.norm(A_speed_soft, axis=1)
            slack_coeff = -np.ones(qp.n_speed_soft_rows, dtype=float)
            if bool(args.speed_row_normalize):
                row_norms_safe = np.maximum(row_norms_soft, float(args.speed_row_norm_eps))
                A_speed_soft = A_speed_soft / row_norms_safe[:, None]
                rhs_speed_soft = rhs_speed_soft / row_norms_safe
                slack_coeff = slack_coeff / row_norms_safe
            A_rebuild_dense[speed_soft_slice, :] = 0.0
            A_rebuild_dense[speed_soft_slice, : qp.n_u] = A_speed_soft
            A_rebuild_dense[speed_soft_slice, int(qp.idx_speed_slack)] = slack_coeff
            l[speed_soft_slice] = -np.inf
            u[speed_soft_slice] = rhs_speed_soft
        elif str(qp.speed_form) == "mean_upper":
            A_speed_soft = np.mean(B_eff_speed, axis=0, keepdims=True)
            rhs_speed_soft = np.asarray([float(qp.speed_upper) - float(np.mean(a_speed))], dtype=float)
            row_norms_soft = np.linalg.norm(A_speed_soft, axis=1)
            slack_coeff = np.asarray([-1.0], dtype=float)
            if bool(args.speed_row_normalize):
                row_norms_safe = np.maximum(row_norms_soft, float(args.speed_row_norm_eps))
                A_speed_soft = A_speed_soft / row_norms_safe[:, None]
                rhs_speed_soft = rhs_speed_soft / row_norms_safe
                slack_coeff = slack_coeff / row_norms_safe
            A_rebuild_dense[speed_soft_slice, :] = 0.0
            A_rebuild_dense[speed_soft_slice, : qp.n_u] = A_speed_soft
            A_rebuild_dense[speed_soft_slice, int(qp.idx_speed_slack)] = slack_coeff
            l[speed_soft_slice] = -np.inf
            u[speed_soft_slice] = rhs_speed_soft
        else:
            raise ValueError(f"unknown speed_form={qp.speed_form!r}")

    if (
        qp.idx_primary_cap is not None
        and qp.n_primary_cap > 0
        and float(primary_cap_rel) > 0.0
        and a_load_primary is not None
        and B_qp_primary is not None
    ):
        H_pri = qp.H_tower
        A_cap_pri = H_pri @ B_qp_primary
        z_free_pri = H_pri @ a_load_primary
        tower_primary_cap_free_rms = float(np.sqrt(np.mean(np.square(z_free_pri)))) if z_free_pri.size else float("nan")
        cap_floor = max(1e-6, float(primary_cap_floor_rel) * max(tower_primary_cap_free_rms, 1e-9))
        tower_primary_cap_floor = float(cap_floor)
        cap_vec = np.maximum(cap_floor, float(primary_cap_rel) * np.abs(z_free_pri))
        cap_slice = slice(int(qp.idx_primary_cap), int(qp.idx_primary_cap + qp.n_primary_cap))
        l[cap_slice] = -cap_vec - z_free_pri
        u[cap_slice] = +cap_vec - z_free_pri
        if A_rebuild_dense is None:
            A_rebuild_dense = qp.A_dense_base.copy()
        A_rebuild_dense[cap_slice, :] = 0.0
        A_rebuild_dense[cap_slice, : qp.n_u] = A_cap_pri

    if (
        qp.idx_secondary_cap is not None
        and qp.n_secondary_cap > 0
        and float(secondary_cap_rel) > 0.0
        and a_load_secondary is not None
        and B_qp_secondary is not None
    ):
        H_sec = qp.H_tower
        A_cap = H_sec @ B_qp_secondary
        z_free = H_sec @ a_load_secondary
        cap_floor = max(float(args.secondary_cap_floor), 1e-6)
        cap_vec = np.maximum(cap_floor, float(secondary_cap_rel) * np.abs(z_free))
        cap_slice = slice(int(qp.idx_secondary_cap), int(qp.idx_secondary_cap + qp.n_secondary_cap))
        l[cap_slice] = -cap_vec - z_free
        u[cap_slice] = +cap_vec - z_free
        if A_rebuild_dense is None:
            A_rebuild_dense = qp.A_dense_base.copy()
        A_rebuild_dense[cap_slice, :] = 0.0
        A_rebuild_dense[cap_slice, : qp.n_u] = A_cap

    stats = {
        "speed_guard_rhs_min": float(speed_guard_rhs_min),
        "speed_guard_row_norm_mean": float(speed_guard_row_norm_mean),
        "speed_guard_row_norm_max": float(speed_guard_row_norm_max),
        "tower_primary_cap_floor": float(tower_primary_cap_floor),
        "tower_primary_cap_free_rms": float(tower_primary_cap_free_rms),
    }
    return l, u, A_rebuild_dense, stats


@dataclass
class QPBundle:
    prob: osqp.OSQP
    P: sp.csc_matrix
    A: sp.csc_matrix
    A_dense_base: np.ndarray
    Ax_idx_secondary_cap: Optional[np.ndarray]
    n_u: int
    n_var: int
    base_l: np.ndarray
    base_u: np.ndarray
    move_blocks: list[tuple[int, int]]
    # Indices into l/u that depend on u_prev (rate first row) and power rhs.
    idx_rate0: int
    idx_power: Optional[int]
    idx_speed: Optional[int]
    n_speed_rows: int
    idx_speed_soft: Optional[int]
    n_speed_soft_rows: int
    idx_speed_slack: Optional[int]
    idx_primary_cap: Optional[int]
    n_primary_cap: int
    idx_secondary_cap: Optional[int]
    n_secondary_cap: int
    # Precomputed pieces for q update:
    M_tower: np.ndarray  # (N, N) = H^T H for tower band
    H_tower: np.ndarray
    load_rel_1p: float
    load_rel_3p: float
    load_1p_halfwidth_hz: float
    load_3p_halfwidth_hz: float
    load_1p_taps: int
    load_3p_taps: int
    load_freq_mode: str  # 'ref' or 'meas'
    load_q_mode: str  # 'full' or 'tower_only'
    f_1p_ref_hz: float
    # Simple cache for measured-frequency mode (quantized 1P).
    f_1p_cache_hz: float
    M_cache: Optional[np.ndarray]
    P_const: np.ndarray  # (N, N) constant Hessian pieces (u + du penalties)
    P_abs_u: np.ndarray  # absolute-u quadratic penalties before delta-coordinate transform
    w_load: float
    Af_load: np.ndarray  # (N, n_f)
    B_blocks_load: list[np.ndarray]  # list of (N, N) lower-triangular blocks
    u_mode_load: str
    u_sched_ref_load: Optional[dict]
    decision_coords: str
    C_u: np.ndarray
    ones_u: np.ndarray
    # u / du weights
    w_u: float
    w_du_chain: float
    w_du_anchor: float
    D_chain: np.ndarray
    anchor_vec: np.ndarray
    ts: float
    # Power constraint model pieces:
    power_c: Optional[np.ndarray]  # (N,)
    Af_pwr: Optional[np.ndarray]   # (N, n_f)
    B_blocks_pwr: Optional[list[np.ndarray]]
    u_mode_pwr: Optional[str]
    u_sched_ref_pwr: Optional[dict]
    # Speed guard model pieces:
    Af_speed: Optional[np.ndarray]
    B_blocks_speed: Optional[list[np.ndarray]]
    u_mode_speed: Optional[str]
    u_sched_ref_speed: Optional[dict]
    speed_form: Optional[str]
    speed_upper: Optional[float]
    speed_target: Optional[str]
    speed_soft_weight: float
    power_c_abs: Optional[np.ndarray]


def _build_qp(
    *,
    N: int,
    ts: float,
    move_blocking_spec: str,
    u_max: float,
    du_max: float,
    w_u: float,
    w_du: float,
    w_du_chain: float,
    w_du_anchor: float,
    decision_coords: str,
    w_u_freq: float,
    u_freq_centers_hz: List[float],
    u_freq_halfwidth_hz: float,
    u_freq_taps: int,
    load_B0: np.ndarray,
    load_Af: np.ndarray,
    load_B_blocks: list[np.ndarray],
    load_u_mode: str,
    load_u_sched_ref: Optional[dict],
    bp_center_hz: float,
    bp_halfwidth_hz: float,
    bp_taps: int,
    w_load: float,
    w_load_mode: str,
    w_load_auto_min: float,
    w_load_auto_max: float,
    load_rel_1p: float,
    load_rel_3p: float,
    load_1p_halfwidth_hz: float,
    load_3p_halfwidth_hz: float,
    load_1p_taps: int,
    load_3p_taps: int,
    load_freq_mode: str,
    load_q_mode: str,
    tower_metric_mode: str,
    tower_fatigue_alpha0: float,
    tower_fatigue_alpha2: float,
    tower_fatigue_alpha4: float,
    tower_metric_normalize: str,
    tower_cone_alpha_band: float,
    tower_cone_alpha_m2: float,
    tower_cone_alpha_m4: float,
    tower_cone_alpha_env70: float,
    tower_cone_alpha_env85: float,
    tower_cone_alpha_env95: float,
    tower_cone_alpha_envlp85: float,
    tower_cone_alpha_envlp95: float,
    tower_cone_alpha_tp90: float,
    tower_cone_alpha_tp95: float,
    tower_cone_alpha_sb1p: float,
    tower_cone_alpha_sb3p: float,
    tower_cone_alpha_sb1p_env95: float,
    tower_cone_alpha_sb3p_env95: float,
    tower_cone_env_floor: float,
    tower_cone_env_sharpness: float,
    tower_cone_lowfreq_cutoff_hz: float,
    tower_cone_tp_window_s: float,
    tower_cone_basis_normalize: str,
    f_1p_ref_hz: float,
    eq_dc: bool,
    eq_lowfreq_hz: List[float],
    power_enable: bool,
    power_eps: float,
    power_B0: Optional[np.ndarray],
    power_Af: Optional[np.ndarray],
    power_B_blocks: Optional[list[np.ndarray]],
    power_u_mode: Optional[str],
    power_u_sched_ref: Optional[dict],
    speed_enable: bool,
    speed_form: str,
    speed_upper: float,
    speed_target: str,
    speed_soft_enable: bool,
    speed_soft_weight: float,
    speed_B0: Optional[np.ndarray],
    speed_Af: Optional[np.ndarray],
    speed_B_blocks: Optional[list[np.ndarray]],
    speed_u_mode: Optional[str],
    speed_u_sched_ref: Optional[dict],
    secondary_cap_enable: bool,
    primary_cap_enable: bool,
) -> QPBundle:
    n_u = int(N)
    n_var = int(N) + (1 if bool(speed_soft_enable) else 0)
    idx_speed_slack = int(N) if bool(speed_soft_enable) else None
    decision_coords = str(decision_coords)
    if decision_coords not in {"absolute_u", "delta_u"}:
        raise ValueError(f"unsupported decision_coords={decision_coords!r}")
    move_blocks = _parse_move_blocking_spec(str(move_blocking_spec), N=n_u)
    C_u = _cumulative_sum_matrix(n_u)
    ones_u = np.ones(n_u, dtype=float)
    fs = 1.0 / float(ts)
    H_tower = _bandpass_toeplitz(
        N=N,
        ts=float(ts),
        center_hz=float(bp_center_hz),
        halfwidth_hz=float(bp_halfwidth_hz),
        taps=int(bp_taps),
    )
    M_tower_band = _bandpass_energy_matrix(
        N=N,
        ts=float(ts),
        center_hz=float(bp_center_hz),
        halfwidth_hz=float(bp_halfwidth_hz),
        taps=int(bp_taps),
    )
    tower_metric_mode = str(tower_metric_mode)
    if tower_metric_mode in ("bandpass", "pafs_online", "fatigue_hybrid_env", "psd_cone"):
        M_tower = M_tower_band
    elif tower_metric_mode == "identity":
        M_tower = np.eye(N, dtype=float)
    else:
        if tower_metric_mode == "fatigue_m4":
            alpha0 = 0.0
            alpha2 = 0.0
            alpha4 = 1.0
        elif tower_metric_mode == "fatigue_fit_abs":
            alpha0 = 0.0
            alpha2 = 2.765487e-02
            alpha4 = 1.049120e-02
        elif tower_metric_mode == "fatigue_custom":
            alpha0 = float(tower_fatigue_alpha0)
            alpha2 = float(tower_fatigue_alpha2)
            alpha4 = float(tower_fatigue_alpha4)
        else:
            raise ValueError(f"unknown tower_metric_mode={tower_metric_mode!r}")
        M0 = float(ts) * M_tower_band
        D1 = _first_diff_energy_matrix(N=N, ts=float(ts))
        D2 = _second_diff_energy_matrix(N=N, ts=float(ts))
        M2 = H_tower.T @ D1.T @ D1 @ H_tower
        M4 = H_tower.T @ D2.T @ D2 @ H_tower
        M_fat = alpha0 * M0 + alpha2 * M2 + alpha4 * M4
        M_tower = _normalize_metric_matrix(M_ref=M_tower_band, M_new=M_fat, mode=str(tower_metric_normalize))

    # Objective terms:
    # Load band-pass energy (multi-band): (a + B u)^T M_total (a + B u)
    B = load_B0

    # Absolute-u quadratic penalties (magnitude + selected frequency bands). When the
    # decision coordinates are delta-u, these penalties are mapped through u = u_prev*1 + C v.
    P_u = 2.0 * float(w_u) * np.eye(n_u)

    # du penalty: split internal horizon smoothing from first-move anchoring.
    D_rate = _diff_matrix(n_u)
    D_chain = _diff_chain_matrix(n_u)
    anchor_vec = np.zeros(n_u, dtype=float)
    if n_u > 0:
        anchor_vec[0] = 1.0
    if decision_coords == "delta_u":
        P_du_chain = np.zeros((n_u, n_u), dtype=float)
        if n_u > 1:
            P_du_chain[1:, 1:] = 2.0 * float(w_du_chain) * np.eye(n_u - 1, dtype=float)
        P_du_anchor = 2.0 * float(w_du_anchor) * np.outer(anchor_vec, anchor_vec)
    else:
        P_du_chain = 2.0 * float(w_du_chain) * (D_chain.T @ D_chain) if D_chain.size else np.zeros((n_u, n_u), dtype=float)
        P_du_anchor = 2.0 * float(w_du_anchor) * np.outer(anchor_vec, anchor_vec)

    # Optional frequency-shaped u penalty (QP-convex): discourage certain bands in the enhancement layer.
    # Example use: penalize u energy near 1P/3P to avoid interfering with rotor-synchronous content.
    P_ufreq = np.zeros((n_u, n_u), dtype=float)
    if float(w_u_freq) > 0.0 and u_freq_centers_hz:
        taps_u = int(u_freq_taps)
        if taps_u % 2 == 0:
            taps_u += 1
        if taps_u > n_u:
            taps_u = n_u if (n_u % 2 == 1) else (n_u - 1)
        half_u = float(u_freq_halfwidth_hz)
        for c in u_freq_centers_hz:
            c = float(c)
            f_u_lo = max(1e-6, c - half_u)
            f_u_hi = c + half_u
            if f_u_hi >= (0.5 * fs):
                continue
            h_u = firwin(taps_u, [f_u_lo, f_u_hi], pass_zero=False, fs=fs)
            H_u = _fir_to_toeplitz(h_u, n_u)
            M_u = H_u.T @ H_u
            P_ufreq += (2.0 * float(w_u_freq)) * M_u

    P_abs_u = P_u + P_ufreq
    if decision_coords == "delta_u":
        P_u_geom = C_u.T @ P_abs_u @ C_u
    else:
        P_u_geom = P_abs_u
    P_const = _embed_u_matrix(P_u_geom + P_du_chain + P_du_anchor, n_var=n_var)
    if bool(speed_soft_enable) and idx_speed_slack is not None and float(speed_soft_weight) > 0.0:
        P_const[int(idx_speed_slack), int(idx_speed_slack)] = 2.0 * float(speed_soft_weight)

    # Multi-band load energy matrix for initial setup (use reference 1P/3P centers).
    load_rel_1p = float(load_rel_1p)
    load_rel_3p = float(load_rel_3p)
    if load_rel_1p < 0 or load_rel_3p < 0:
        raise ValueError("load_rel_1p/load_rel_3p must be nonnegative")
    f_1p_ref_hz = float(f_1p_ref_hz)
    if not math.isfinite(f_1p_ref_hz) or f_1p_ref_hz <= 0:
        # Safe fallback: no 1P/3P shaping if ref is unknown.
        load_rel_1p = 0.0
        load_rel_3p = 0.0
        f_1p_ref_hz = float("nan")

    M_total = M_tower
    if (load_rel_1p > 0.0) or (load_rel_3p > 0.0):
        M_1p = _bandpass_energy_matrix(
            N=N,
            ts=float(ts),
            center_hz=f_1p_ref_hz,
            halfwidth_hz=float(load_1p_halfwidth_hz),
            taps=int(load_1p_taps),
        )
        M_3p = _bandpass_energy_matrix(
            N=N,
            ts=float(ts),
            center_hz=3.0 * f_1p_ref_hz,
            halfwidth_hz=float(load_3p_halfwidth_hz),
            taps=int(load_3p_taps),
        )
        M_total = M_total + load_rel_1p * M_1p + load_rel_3p * M_3p

    # Load Hessian unit for auto scaling.
    Q_unit_abs = 2.0 * (B.T @ M_total @ B)
    Q_unit = C_u.T @ Q_unit_abs @ C_u if decision_coords == "delta_u" else Q_unit_abs

    w_load_eff = float(w_load)
    w_load_mode = str(w_load_mode)
    if w_load_mode != "fixed":
        diag_Q = np.diag(Q_unit)
        diag_P = np.diag(P_const)
        eps = 1e-18
        if w_load_mode == "auto_mean_diag":
            denom = float(np.mean(diag_Q))
            numer = float(np.mean(diag_P))
            w_load_eff = numer / max(denom, eps)
        elif w_load_mode == "auto_trace":
            denom = float(np.trace(Q_unit))
            numer = float(np.trace(P_const))
            w_load_eff = numer / max(denom, eps)
        elif w_load_mode == "auto_max_diag":
            denom = float(np.max(diag_Q))
            numer = float(np.max(diag_P))
            w_load_eff = numer / max(denom, eps)
        else:
            raise ValueError(f"unknown w_load_mode={w_load_mode!r}")

        w_min = float(w_load_auto_min)
        w_max = float(w_load_auto_max)
        if not math.isfinite(w_min) or w_min < 0:
            w_min = 0.0
        if not math.isfinite(w_max) or w_max <= 0:
            w_max = float("inf")
        w_load_eff = float(np.clip(w_load_eff, w_min, w_max))

    Q_load = float(w_load_eff) * Q_unit  # OSQP P contribution
    P = _embed_u_matrix(Q_load, n_var=n_var) + P_const

    # Constraints:
    A_blocks = []
    l_blocks = []
    u_blocks = []

    # 1) amplitude bounds
    A_u = np.zeros((n_u, n_var), dtype=float)
    if decision_coords == "delta_u":
        A_u[:, :n_u] = C_u
    else:
        A_u[:, :n_u] = np.eye(n_u)
    A_blocks.append(A_u)
    l_blocks.append(-float(u_max) * np.ones(n_u))
    u_blocks.append(+float(u_max) * np.ones(n_u))

    # 2) rate bounds (first row depends on u_prev)
    A_du = np.zeros((n_u, n_var), dtype=float)
    if decision_coords == "delta_u":
        A_du[:, :n_u] = np.eye(n_u, dtype=float)
    else:
        A_du[:, :n_u] = D_rate
    A_blocks.append(A_du)
    # placeholder; we will overwrite first row each step
    l_du = -float(du_max) * np.ones(n_u)
    u_du = +float(du_max) * np.ones(n_u)
    l_blocks.append(l_du)
    u_blocks.append(u_du)
    idx_rate0 = sum(len(x) for x in l_blocks[:-1])  # first row index within stacked l/u for A_du

    # 3) equality constraints (DC and lowfreq)
    eq_rows = []
    if eq_dc:
        eq_rows.append(np.ones(n_u, dtype=float) / float(n_u))
    t_grid = np.arange(n_u, dtype=float) * float(ts)
    for f in eq_lowfreq_hz:
        f = float(f)
        if f <= 0:
            continue
        eq_rows.append(np.cos(2.0 * math.pi * f * t_grid) / float(n_u))
        eq_rows.append(np.sin(2.0 * math.pi * f * t_grid) / float(n_u))
    if eq_rows:
        A_eq_u = np.vstack(eq_rows)
        A_eq = np.zeros((A_eq_u.shape[0], n_var), dtype=float)
        A_eq[:, :n_u] = A_eq_u
        A_blocks.append(A_eq)
        l_blocks.append(np.zeros(A_eq.shape[0]))
        u_blocks.append(np.zeros(A_eq.shape[0]))

    A_blk_u = _move_blocking_equalities(move_blocks, N=n_u)
    if A_blk_u.shape[0] > 0:
        A_blk = np.zeros((A_blk_u.shape[0], n_var), dtype=float)
        A_blk[:, :n_u] = A_blk_u
        A_blocks.append(A_blk)
        l_blocks.append(np.zeros(A_blk.shape[0], dtype=float))
        u_blocks.append(np.zeros(A_blk.shape[0], dtype=float))

    # 4) mean power constraint: mean(P_pred) >= (1-eps)*P_meas
    idx_power = None
    power_c = None
    power_c_abs = None
    if power_enable:
        if power_B0 is None or power_Af is None:
            raise ValueError("power_enable requires power model matrices")
        power_c_abs = (power_B0.mean(axis=0)).astype(float)  # (N,)
        power_c = (power_c_abs @ C_u).astype(float) if decision_coords == "delta_u" else power_c_abs
        A_pwr = np.zeros((1, n_var), dtype=float)
        A_pwr[0, :n_u] = power_c
        A_blocks.append(A_pwr)
        # placeholder RHS; updated each step.
        l_blocks.append(np.array([0.0], dtype=float))
        u_blocks.append(np.array([np.inf], dtype=float))
        idx_power = sum(len(x) for x in l_blocks[:-1])

    idx_primary_cap = None
    n_primary_cap = 0
    idx_secondary_cap = None
    n_secondary_cap = 0
    idx_speed = None
    n_speed_rows = 0
    idx_speed_soft = None
    n_speed_soft_rows = 0
    if bool(speed_enable):
        if speed_B0 is None or speed_Af is None or speed_B_blocks is None or speed_u_mode is None:
            raise ValueError("speed_enable requires speed model matrices")
        idx_speed = sum(len(x) for x in l_blocks)
        if str(speed_form) == "step_upper":
            n_speed_rows = int(n_u)
            A_speed = np.zeros((int(n_u), int(n_var)), dtype=float)
            A_speed[:, :n_u] = np.full((int(n_u), int(n_u)), 1e-12, dtype=float)
            A_blocks.append(A_speed)
            l_blocks.append(-1e6 * np.ones(int(n_u), dtype=float))
            u_blocks.append(+1e6 * np.ones(int(n_u), dtype=float))
        elif str(speed_form) == "mean_upper":
            n_speed_rows = 1
            A_speed = np.zeros((1, int(n_var)), dtype=float)
            A_speed[0, :n_u] = np.full((1, int(n_u)), 1e-12, dtype=float)
            A_blocks.append(A_speed)
            l_blocks.append(np.array([-1e6], dtype=float))
            u_blocks.append(np.array([+1e6], dtype=float))
        else:
            raise ValueError(f"unknown speed_form={speed_form!r}")
    if bool(speed_soft_enable):
        if speed_B0 is None or speed_Af is None or speed_B_blocks is None or speed_u_mode is None:
            raise ValueError("speed_soft_enable requires speed model matrices")
        if idx_speed_slack is None:
            raise ValueError("speed_soft_enable requires a valid slack index")
        idx_speed_soft = sum(len(x) for x in l_blocks)
        if str(speed_form) == "step_upper":
            n_speed_soft_rows = int(n_u)
            A_speed_soft = np.zeros((int(n_u), int(n_var)), dtype=float)
            A_speed_soft[:, :n_u] = np.full((int(n_u), int(n_u)), 1e-12, dtype=float)
            A_speed_soft[:, int(idx_speed_slack)] = -1.0
            A_blocks.append(A_speed_soft)
            l_blocks.append(-1e6 * np.ones(int(n_u), dtype=float))
            u_blocks.append(+1e6 * np.ones(int(n_u), dtype=float))
        elif str(speed_form) == "mean_upper":
            n_speed_soft_rows = 1
            A_speed_soft = np.zeros((1, int(n_var)), dtype=float)
            A_speed_soft[0, :n_u] = np.full((1, int(n_u)), 1e-12, dtype=float)
            A_speed_soft[0, int(idx_speed_slack)] = -1.0
            A_blocks.append(A_speed_soft)
            l_blocks.append(np.array([-1e6], dtype=float))
            u_blocks.append(np.array([+1e6], dtype=float))
        else:
            raise ValueError(f"unknown speed_form={speed_form!r}")
        A_slack = np.zeros((1, int(n_var)), dtype=float)
        A_slack[0, int(idx_speed_slack)] = 1.0
        A_blocks.append(A_slack)
        l_blocks.append(np.array([0.0], dtype=float))
        u_blocks.append(np.array([np.inf], dtype=float))
    if bool(primary_cap_enable):
        idx_primary_cap = sum(len(x) for x in l_blocks)
        n_primary_cap = int(n_u)
        A_cap_primary = np.zeros((int(n_u), int(n_var)), dtype=float)
        A_cap_primary[:, :n_u] = np.full((int(n_u), int(n_u)), 1e-12, dtype=float)
        A_blocks.append(A_cap_primary)
        l_blocks.append(-1e6 * np.ones(int(n_u), dtype=float))
        u_blocks.append(+1e6 * np.ones(int(n_u), dtype=float))
    if bool(secondary_cap_enable):
        idx_secondary_cap = sum(len(x) for x in l_blocks)
        n_secondary_cap = int(n_u)
        A_cap = np.zeros((int(n_u), int(n_var)), dtype=float)
        A_cap[:, :n_u] = np.full((int(n_u), int(n_u)), 1e-12, dtype=float)
        A_blocks.append(A_cap)
        l_blocks.append(-1e6 * np.ones(int(n_u), dtype=float))
        u_blocks.append(+1e6 * np.ones(int(n_u), dtype=float))

    A = np.vstack(A_blocks)
    l0 = np.concatenate(l_blocks)
    u0 = np.concatenate(u_blocks)

    # OSQP setup
    # OSQP expects P in upper-triangular form (CSC). Keep a fixed pattern so we can update Px safely.
    triu_indices, triu_indptr = _csc_triu_indices(n_var)
    triu_data = _csc_triu_data(P)
    P_sp = sp.csc_matrix((triu_data, triu_indices, triu_indptr), shape=(n_var, n_var))
    A_sp = sp.csc_matrix(A)
    ax_idx_secondary_cap = None
    if idx_secondary_cap is not None and n_secondary_cap > 0:
        cap_row_lo = int(idx_secondary_cap)
        cap_row_hi = int(idx_secondary_cap + n_secondary_cap)
        ax_idx_list: list[int] = []
        for j in range(A_sp.shape[1]):
            start = int(A_sp.indptr[j])
            end = int(A_sp.indptr[j + 1])
            rows = A_sp.indices[start:end]
            for off, row in enumerate(rows):
                if cap_row_lo <= int(row) < cap_row_hi:
                    ax_idx_list.append(start + off)
        ax_idx_secondary_cap = np.asarray(ax_idx_list, dtype=np.int32)
        expected = int(n_secondary_cap) * int(n_u)
        if ax_idx_secondary_cap.size != expected:
            raise ValueError(
                f"secondary-cap Ax index size mismatch: got {ax_idx_secondary_cap.size}, expected {expected}"
            )
    prob = osqp.OSQP()
    prob.setup(P=P_sp, q=np.zeros(n_var), A=A_sp, l=l0, u=u0, verbose=False, polish=False, warm_start=True)

    return QPBundle(
        prob=prob,
        P=P_sp,
        A=A_sp,
        A_dense_base=A.copy(),
        Ax_idx_secondary_cap=ax_idx_secondary_cap,
        n_u=n_u,
        n_var=n_var,
        base_l=l0,
        base_u=u0,
        move_blocks=move_blocks,
        idx_rate0=idx_rate0,
        idx_power=idx_power,
        idx_speed=idx_speed,
        n_speed_rows=n_speed_rows,
        idx_speed_soft=idx_speed_soft,
        n_speed_soft_rows=n_speed_soft_rows,
        idx_speed_slack=idx_speed_slack,
        idx_primary_cap=idx_primary_cap,
        n_primary_cap=n_primary_cap,
        idx_secondary_cap=idx_secondary_cap,
        n_secondary_cap=n_secondary_cap,
        M_tower=M_tower,
        H_tower=H_tower,
        load_rel_1p=float(load_rel_1p),
        load_rel_3p=float(load_rel_3p),
        load_1p_halfwidth_hz=float(load_1p_halfwidth_hz),
        load_3p_halfwidth_hz=float(load_3p_halfwidth_hz),
        load_1p_taps=int(load_1p_taps),
        load_3p_taps=int(load_3p_taps),
        load_freq_mode=str(load_freq_mode),
        load_q_mode=str(load_q_mode),
        f_1p_ref_hz=float(f_1p_ref_hz) if math.isfinite(f_1p_ref_hz) else float("nan"),
        f_1p_cache_hz=float(f_1p_ref_hz) if (str(load_freq_mode) == "ref" and math.isfinite(f_1p_ref_hz) and (load_rel_1p > 0.0 or load_rel_3p > 0.0)) else float("nan"),
        M_cache=M_total if (str(load_freq_mode) == "ref" and math.isfinite(f_1p_ref_hz) and (load_rel_1p > 0.0 or load_rel_3p > 0.0)) else None,
        P_const=P_const,
        P_abs_u=P_abs_u,
        w_load=float(w_load_eff),
        Af_load=load_Af,
        B_blocks_load=load_B_blocks,
        u_mode_load=str(load_u_mode),
        u_sched_ref_load=load_u_sched_ref,
        decision_coords=str(decision_coords),
        C_u=C_u,
        ones_u=ones_u,
        w_u=float(w_u),
        w_du_chain=float(w_du_chain),
        w_du_anchor=float(w_du_anchor),
        D_chain=D_chain,
        anchor_vec=anchor_vec,
        ts=float(ts),
        power_c=power_c,
        power_c_abs=power_c_abs,
        Af_pwr=power_Af if power_enable else None,
        B_blocks_pwr=power_B_blocks if power_enable else None,
        u_mode_pwr=str(power_u_mode) if power_enable and power_u_mode is not None else None,
        u_sched_ref_pwr=power_u_sched_ref if power_enable else None,
        Af_speed=speed_Af if (speed_enable or speed_soft_enable) else None,
        B_blocks_speed=speed_B_blocks if (speed_enable or speed_soft_enable) else None,
        u_mode_speed=str(speed_u_mode) if (speed_enable or speed_soft_enable) and speed_u_mode is not None else None,
        u_sched_ref_speed=speed_u_sched_ref if (speed_enable or speed_soft_enable) else None,
        speed_form=str(speed_form) if (speed_enable or speed_soft_enable) else None,
        speed_upper=float(speed_upper) if (speed_enable or speed_soft_enable) else None,
        speed_target=str(speed_target) if (speed_enable or speed_soft_enable) else None,
        speed_soft_weight=float(speed_soft_weight),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="ipc:///tmp/rosco_wfc")
    ap.add_argument(
        "--interface",
        type=Path,
        default=(_REPO_ROOT / "external" / "rosco-src" / "rosco" / "controller" / "rosco_registry" / "wfc_interface.yaml"),
    )
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--wind-mps", type=int, required=True, help="Select model set by wind speed (ws).")
    ap.add_argument("--objective", choices=["acc_bp", "myt_bp", "tower_env_l2", "tower_proxy_bp", "tower_latent_bp", "tower_latent2_bp", "yaw_bp"], default="acc_bp")
    ap.add_argument("--secondary-objective", choices=["none", "acc_bp", "myt_bp", "tower_env_l2", "tower_proxy_bp", "tower_latent_bp", "tower_latent2_bp", "yaw_bp"], default="none")
    ap.add_argument("--secondary-rel", type=float, default=0.0, help="Relative weight on optional secondary load objective.")
    ap.add_argument(
        "--secondary-form",
        choices=["full", "hessian_only", "q_only"],
        default="full",
        help="How the secondary objective enters the QP: full adds Hessian+q, hessian_only keeps only the injected-energy guard, q_only keeps only the affine term.",
    )
    ap.add_argument(
        "--primary-form",
        choices=["full", "hessian_only", "q_shrink", "q_only_a_blend"],
        default="full",
        help="How the primary objective enters the blended QP: full uses blended Hessian+q; hessian_only keeps blended Hessian but uses the primary-model affine term; q_shrink interpolates the affine term.",
    )
    ap.add_argument("--primary-q-shrink-mode", choices=["none", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v", "local_opt_gamma", "local_opt_gamma_kkt", "local_opt_speed", "local_opt_speed_project", "local_opt_value_speed"], default="none")
    ap.add_argument("--primary-q-shrink-center", type=float, default=0.0)
    ap.add_argument("--primary-q-shrink-width", type=float, default=1.0)
    ap.add_argument("--primary-q-shrink-lo", type=float, default=0.0)
    ap.add_argument("--primary-q-shrink-hi", type=float, default=1.0)
    ap.add_argument("--primary-q-shrink-rbf-table", type=str, default="", help="Semicolon-separated wind,beta_deg,lambda anchors for primary q shrink.")
    ap.add_argument("--primary-q-shrink-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--primary-q-shrink-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--primary-q-speed-form", choices=["step_upper", "mean_upper"], default="step_upper")
    ap.add_argument("--primary-q-speed-upper", type=float, default=float("nan"))
    ap.add_argument("--primary-q-speed-weight", type=float, default=25.0)
    ap.add_argument("--primary-q-speed-grid", type=int, default=25)
    ap.add_argument("--primary-q-value-speed-weight", type=float, default=100.0)
    ap.add_argument("--primary-q-value-trust-weight", type=float, default=0.0)
    ap.add_argument(
        "--primary-a-blend-mode",
        choices=["none", "match_gamma", "scale_gamma", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v"],
        default="none",
    )
    ap.add_argument("--primary-a-blend-scale", type=float, default=1.0)
    ap.add_argument("--primary-a-blend-center", type=float, default=0.0)
    ap.add_argument("--primary-a-blend-width", type=float, default=1.0)
    ap.add_argument("--primary-a-blend-lo", type=float, default=0.0)
    ap.add_argument("--primary-a-blend-hi", type=float, default=1.0)
    ap.add_argument("--primary-a-blend-rbf-table", type=str, default="")
    ap.add_argument("--primary-a-blend-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--primary-a-blend-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--secondary-cap-rel", type=float, default=0.0, help="Optional relative band-pass cap on the secondary prediction; 0 disables.")
    ap.add_argument("--secondary-cap-floor", type=float, default=0.05, help="Lower floor for the normalized secondary cap envelope.")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--ts", type=float, default=0.1)
    ap.add_argument(
        "--move-blocking-spec",
        type=str,
        default="",
        help="Exact piecewise-constant move blocking in countxsteps form, e.g. '20x1,6x5,5x30'. Empty disables blocking.",
    )
    ap.add_argument("--u-max-deg", type=float, default=0.5)
    ap.add_argument("--du-max-deg-s", type=float, default=7.5)
    ap.add_argument("--w-load", type=float, default=1.0)
    ap.add_argument(
        "--w-load-mode",
        choices=["fixed", "auto_mean_diag", "auto_trace", "auto_max_diag"],
        default="fixed",
        help="How to set w_load: fixed uses --w-load, auto_* matches load Hessian curvature to u/du regularizers.",
    )
    ap.add_argument("--w-load-auto-min", type=float, default=0.0, help="Lower clip for auto w_load.")
    ap.add_argument("--w-load-auto-max", type=float, default=1e3, help="Upper clip for auto w_load.")
    ap.add_argument("--load-1p-rel", type=float, default=0.0, help="Relative load penalty weight for 1P band (added to tower band).")
    ap.add_argument("--load-3p-rel", type=float, default=0.0, help="Relative load penalty weight for 3P band (added to tower band).")
    ap.add_argument("--load-1p-halfwidth-hz", type=float, default=0.02)
    ap.add_argument("--load-3p-halfwidth-hz", type=float, default=0.05)
    ap.add_argument("--load-1p-taps", type=int, default=21)
    ap.add_argument("--load-3p-taps", type=int, default=21)
    ap.add_argument(
        "--load-freq-mode",
        choices=["ref", "meas"],
        default="ref",
        help="How to choose 1P/3P band centers: ref uses model omega_ref, meas uses measured RotSpeed each step.",
    )
    ap.add_argument(
        "--load-q-mode",
        choices=["full", "tower_only"],
        default="full",
        help="How to form the load linear term q: full uses all bands; tower_only penalizes 1P/3P only via Hessian (discourage injection).",
    )
    ap.add_argument(
        "--load-1p-ref-hz",
        type=float,
        default=float("nan"),
        help="Override 1P center frequency (Hz) for load multi-band penalty. If unset, uses model u_sched_ref.omega_ref.",
    )
    ap.add_argument("--w-u", type=float, default=0.05)
    ap.add_argument("--w-du", type=float, default=0.2)
    ap.add_argument("--w-du-chain", type=float, default=float("nan"))
    ap.add_argument("--w-du-anchor", type=float, default=float("nan"))
    ap.add_argument("--decision-coords", choices=["absolute_u", "delta_u"], default="absolute_u")
    ap.add_argument("--u-prev-ref-mode", choices=["raw", "avg2", "ema"], default="raw")
    ap.add_argument("--u-prev-ref-alpha", type=float, default=0.5)
    ap.add_argument("--u-prev-ref-deadband-rad", type=float, default=0.0)
    ap.add_argument(
        "--act-limit-scale-mode",
        choices=["none", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v"],
        default="none",
        help="Optional runtime schedule that scales both u_max and du_max.",
    )
    ap.add_argument("--act-limit-scale-center", type=float, default=0.0)
    ap.add_argument("--act-limit-scale-width", type=float, default=1.0)
    ap.add_argument("--act-limit-scale-lo", type=float, default=1.0)
    ap.add_argument("--act-limit-scale-hi", type=float, default=1.0)
    ap.add_argument("--act-limit-scale-rbf-table", type=str, default="", help="Semicolon-separated wind,beta_deg,scale anchors for actuator-limit scheduling.")
    ap.add_argument("--act-limit-scale-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--act-limit-scale-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument(
        "--w-u-freq",
        type=float,
        default=0.0,
        help="Weight on band-pass energy of u in specified frequency bands (QP-convex).",
    )
    ap.add_argument(
        "--u-freq-centers-hz",
        type=str,
        default="",
        help="Comma-separated u band-pass centers (Hz), e.g. '0.2,0.6'.",
    )
    ap.add_argument("--u-freq-halfwidth-hz", type=float, default=0.02)
    ap.add_argument("--u-freq-taps", type=int, default=21)
    ap.add_argument("--bp-center-hz", type=float, default=0.327724)
    ap.add_argument("--bp-halfwidth-hz", type=float, default=0.03)
    ap.add_argument("--bp-taps", type=int, default=11)
    ap.add_argument(
        "--tower-metric-mode",
        choices=["bandpass", "identity", "fatigue_m4", "fatigue_fit_abs", "fatigue_custom", "pafs_online", "fatigue_hybrid_env", "psd_cone"],
        default="bandpass",
        help="Primary tower quadratic metric: baseline band-pass energy or fatigue-aware spectral-moment surrogate.",
    )
    ap.add_argument("--tower-fatigue-alpha0", type=float, default=1.0)
    ap.add_argument("--tower-fatigue-alpha2", type=float, default=0.0)
    ap.add_argument("--tower-fatigue-alpha4", type=float, default=0.0)
    ap.add_argument("--tower-fatigue-alphae", type=float, default=0.0)
    ap.add_argument("--tower-pafs-alpha", type=float, default=0.2)
    ap.add_argument("--tower-pafs-mw", type=float, default=6.0)
    ap.add_argument("--tower-pafs-eps", type=float, default=1e-8)
    ap.add_argument(
        "--tower-metric-normalize",
        choices=["none", "trace", "mean_diag", "max_diag"],
        default="trace",
        help="How to normalize fatigue-aware M_tower back to the baseline band-pass scale.",
    )
    ap.add_argument("--tower-cone-alpha-band", type=float, default=1.0)
    ap.add_argument("--tower-cone-alpha-m2", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-m4", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-env70", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-env85", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-env95", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-envlp85", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-envlp95", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-tp90", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-tp95", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-sb1p", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-sb3p", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-sb1p-env95", type=float, default=0.0)
    ap.add_argument("--tower-cone-alpha-sb3p-env95", type=float, default=0.0)
    ap.add_argument("--tower-cone-env-floor", type=float, default=0.2)
    ap.add_argument("--tower-cone-env-sharpness", type=float, default=8.0)
    ap.add_argument("--tower-cone-lowfreq-cutoff-hz", type=float, default=0.08)
    ap.add_argument("--tower-cone-tp-window-s", type=float, default=1.0)
    ap.add_argument(
        "--tower-cone-basis-normalize",
        choices=["none", "trace", "mean_diag", "max_diag"],
        default="trace",
        help="Per-basis normalization before PSD-cone combination.",
    )
    ap.add_argument(
        "--tower-structured-form",
        choices=["none", "guarded_shape", "state_gate"],
        default="none",
        help="Structured tower objective wrapper around psd_cone pieces: guarded_shape enforces a band-energy funnel via multi-QP search; state_gate scales shape correction by envelope/op-point severity.",
    )
    ap.add_argument("--tower-structured-shape-gain", type=float, default=1.0, help="Global gain applied to the non-band shape metric when using a structured tower objective.")
    ap.add_argument("--tower-guard-eps", type=float, default=0.03, help="Allowed relative relaxation of the band-only optimum in guarded_shape mode.")
    ap.add_argument("--tower-guard-mus", type=str, default="0,0.1,0.3,1.0,3.0,10.0", help="Comma-separated band multipliers searched in guarded_shape mode.")
    ap.add_argument("--tower-gate-env-center", type=float, default=1.45, help="Sigmoid center for the dimensionless envelope severity gate.")
    ap.add_argument("--tower-gate-env-width", type=float, default=0.15, help="Sigmoid width for the dimensionless envelope severity gate.")
    ap.add_argument(
        "--tower-gate-op-mode",
        choices=["none", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v"],
        default="none",
        help="Optional slow op-point gate multiplied with the envelope gate in state_gate mode.",
    )
    ap.add_argument("--tower-gate-op-center", type=float, default=0.0)
    ap.add_argument("--tower-gate-op-width", type=float, default=1.0)
    ap.add_argument("--tower-gate-op-lo", type=float, default=0.0)
    ap.add_argument("--tower-gate-op-hi", type=float, default=1.0)
    ap.add_argument("--tower-gate-op-rbf-table", type=str, default="", help="Semicolon-separated wind,beta_deg,gate anchors for state_gate op-point scheduling.")
    ap.add_argument("--tower-gate-op-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--tower-gate-op-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--tower-primary-cap-rel", type=float, default=0.0, help="Relative per-step peak cap on the primary tower band-pass prediction; 0 disables.")
    ap.add_argument("--tower-primary-cap-floor-rel", type=float, default=0.05, help="Minimum cap floor as a fraction of free-response band-pass RMS.")
    ap.add_argument("--tower-irls-iters", type=int, default=0, help="Number of IRLS re-solves applied to the primary tower metric.")
    ap.add_argument("--tower-irls-m", type=float, default=10.0, help="Fatigue exponent used by IRLS reweighting.")
    ap.add_argument("--tower-irls-eps", type=float, default=1e-4, help="Stability epsilon inside IRLS weights.")
    ap.add_argument("--tower-irls-w-clip", type=float, default=100.0, help="Upper clip on normalized IRLS weights.")
    ap.add_argument("--tower-irls-band-rel", type=float, default=1.0, help="Relative replacement strength for the band metric when IRLS is enabled.")
    ap.add_argument("--eq-dc", action="store_true", help="Add DC (mean(u)=0) equality constraint.")
    ap.add_argument("--eq-lowfreq-hz", type=str, default="", help="Comma-separated lowfreqs to project out (Hz).")
    ap.add_argument("--power-enable", action="store_true", help="Enable mean power lower-bound constraint.")
    ap.add_argument("--power-eps", type=float, default=0.005, help="Allowed mean power loss fraction.")
    ap.add_argument("--speed-enable", action="store_true", help="Enable predicted speed upper-bound guard.")
    ap.add_argument("--speed-target", type=str, choices=["RotSpeed", "GenSpeed"], default="RotSpeed")
    ap.add_argument("--speed-models-dir", type=Path, default=None, help="Optional model bank for speed guard; defaults to --models-dir.")
    ap.add_argument("--speed-blend-models-dir", type=Path, default=None, help="Optional blend model bank for speed guard; defaults to --blend-models-dir if present.")
    ap.add_argument("--speed-form", type=str, choices=["step_upper", "mean_upper"], default="step_upper")
    ap.add_argument(
        "--speed-upper",
        type=float,
        default=float("nan"),
        help=(
            "Upper bound on predicted speed target in the trained model's native units. "
            "Current RotSpeed banks in this repo are trained on ROSCO/ZMQ rad/s signals, "
            "while downstream analysis tables often report rpm."
        ),
    )
    ap.add_argument("--speed-row-normalize", dest="speed_row_normalize", action="store_true", help="Row-normalize speed-guard inequalities before solving.")
    ap.add_argument("--no-speed-row-normalize", dest="speed_row_normalize", action="store_false", help="Disable row normalization for speed-guard inequalities.")
    ap.set_defaults(speed_row_normalize=True)
    ap.add_argument("--speed-row-norm-eps", type=float, default=1e-9, help="Minimum row norm used when normalizing speed-guard inequalities.")
    ap.add_argument("--speed-soft-enable", action="store_true", help="Enable a soft upper-envelope speed guard with one shared slack variable.")
    ap.add_argument("--speed-soft-weight", type=float, default=0.0, help="Quadratic weight on the shared speed-guard slack variable.")
    ap.add_argument("--log-csv", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--schedule-tau-s", type=float, default=5.0, help="Low-pass time constant for runtime scheduling signals (HorWindV, BlPitchCMeas).")
    ap.add_argument("--w-load-schedule-mode", choices=["none", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v"], default="none")
    ap.add_argument("--w-load-schedule-center", type=float, default=0.0, help="Center for sigmoid schedule (m/s for sigmoid_v, deg for sigmoid_beta).")
    ap.add_argument("--w-load-schedule-width", type=float, default=1.0, help="Width for sigmoid schedule.")
    ap.add_argument("--w-load-schedule-lo", type=float, default=0.0, help="Low output for sigmoid w_load schedule.")
    ap.add_argument("--w-load-schedule-hi", type=float, default=0.0, help="High output for sigmoid w_load schedule.")
    ap.add_argument("--w-load-schedule-rbf-table", type=str, default="", help="Semicolon-separated 'wind_mps,beta_deg,w_load' anchors for smooth RBF scheduling.")
    ap.add_argument("--w-load-schedule-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--w-load-schedule-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--blend-models-dir", type=Path, default=None, help="Optional secondary model bank to blend with the primary model bank.")
    ap.add_argument("--blend-mode", choices=["none", "sigmoid_v", "sigmoid_beta", "rbf_vbeta", "piecewise_beta", "piecewise_v"], default="none")
    ap.add_argument(
        "--blend-structure",
        choices=["whole_model", "g_only_load"],
        default="whole_model",
        help="whole_model blends both a and G for the load predictor; g_only_load keeps primary a and only blends load G.",
    )
    ap.add_argument("--blend-center", type=float, default=0.0, help="Center for sigmoid gamma schedule (m/s or deg).")
    ap.add_argument("--blend-width", type=float, default=1.0, help="Width for sigmoid gamma schedule.")
    ap.add_argument("--blend-gamma-lo", type=float, default=0.0, help="Low output for gamma schedule.")
    ap.add_argument("--blend-gamma-hi", type=float, default=1.0, help="High output for gamma schedule.")
    ap.add_argument("--blend-rbf-table", type=str, default="", help="Semicolon-separated 'wind_mps,beta_deg,gamma' anchors for smooth model blending.")
    ap.add_argument("--blend-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--blend-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--synth-b-mode", choices=["none", "svd_energy"], default="none")
    ap.add_argument("--synth-b-apply", choices=["hessian_only", "hessian_and_q"], default="hessian_only")
    ap.add_argument("--synth-b-source", choices=["primary", "effective"], default="primary")
    ap.add_argument("--synth-b-energy-retain", type=float, default=0.85)
    ap.add_argument("--synth-b-min-rank", type=int, default=1)
    ap.add_argument("--synth-b-max-rank", type=int, default=0, help="0 means no explicit cap on retained rank.")
    ap.add_argument("--synth-b-gamma-mode", type=str, default="none")
    ap.add_argument("--synth-b-gamma-center", type=float, default=0.0)
    ap.add_argument("--synth-b-gamma-width", type=float, default=1.0)
    ap.add_argument("--synth-b-gamma-lo", type=float, default=0.0)
    ap.add_argument("--synth-b-gamma-hi", type=float, default=1.0)
    ap.add_argument("--synth-b-gamma-rbf-table", type=str, default="", help="Semicolon-separated 'wind_mps,beta_deg,gamma' anchors for synthetic B blending.")
    ap.add_argument("--synth-b-gamma-rbf-v-sigma", type=float, default=1.5)
    ap.add_argument("--synth-b-gamma-rbf-beta-sigma-deg", type=float, default=3.0)
    ap.add_argument("--g-norm-mode", choices=["none", "trace", "mean_diag", "max_diag"], default="none")
    ap.add_argument(
        "--g-norm-form",
        choices=["scale_B", "metric_only"],
        default="scale_B",
        help="How runtime curvature normalization enters the load objective: scale_B rescales B in both Hessian and q; metric_only keeps q on the physical predictor and applies curvature compensation only in the Hessian.",
    )
    ap.add_argument("--g-norm-min", type=float, default=0.25, help="Lower clip for runtime control-kernel normalization scale.")
    ap.add_argument("--g-norm-max", type=float, default=4.0, help="Upper clip for runtime control-kernel normalization scale.")
    ap.add_argument("--g-norm-eps", type=float, default=1e-18, help="Stability epsilon for runtime control-kernel normalization.")
    ap.add_argument("--dry-run", action="store_true", help="Build QP and exit (no ZMQ bind).")
    args = ap.parse_args()

    idef = _load_interface(args.interface)
    expected_n = len(idef.measurements)
    meas_names = idef.measurements

    wind_mps = int(args.wind_mps)
    N = int(args.N)
    ts = float(args.ts)
    if N <= 0:
        raise SystemExit("N must be positive")

    # Load models (fixed for run).
    load_target = _objective_to_target(str(args.objective))
    secondary_rel = max(float(args.secondary_rel), 0.0)
    secondary_cap_rel = max(float(args.secondary_cap_rel), 0.0)
    secondary_target = None
    if str(args.secondary_objective) != "none" and (secondary_rel > 0.0 or secondary_cap_rel > 0.0):
        secondary_target = _objective_to_target(str(args.secondary_objective))
        if str(secondary_target) == str(load_target):
            secondary_target = None
    load_meta, load_thetas = _load_model(args.models_dir, wind_mps=wind_mps, target=load_target)
    power_enable = bool(args.power_enable)
    pwr_meta = None
    pwr_thetas = None
    if power_enable:
        pwr_meta, pwr_thetas = _load_model(args.models_dir, wind_mps=wind_mps, target="GenPwr")
    speed_models_dir = args.speed_models_dir.resolve() if args.speed_models_dir is not None else args.models_dir
    speed_bundle = None
    speed_q_shrink_active = str(args.primary_form) == "q_shrink" and str(args.primary_q_shrink_mode) in ("local_opt_speed", "local_opt_speed_project", "local_opt_value_speed")
    speed_objective_active = bool(args.speed_enable) or bool(args.speed_soft_enable) or bool(speed_q_shrink_active)
    if speed_objective_active:
        speed_q_upper = float(args.primary_q_speed_upper)
        if not math.isfinite(speed_q_upper):
            speed_q_upper = float(args.speed_upper)
        if not math.isfinite(float(args.speed_upper)) and not math.isfinite(speed_q_upper):
            raise SystemExit("--speed-enable/--speed-soft-enable/--primary-q-shrink-mode local_opt_speed requires finite speed upper")
        speed_bundle = _maybe_load_model(speed_models_dir, wind_mps=wind_mps, target=str(args.speed_target))
        if speed_bundle is None:
            raise SystemExit(f"missing speed model target={args.speed_target!r} under {speed_models_dir}")
    feature_names = load_meta["feature_names"]
    pwr_feature_names = pwr_meta["feature_names"] if pwr_meta is not None else feature_names
    n_f = int(load_meta["meta"]["n_f"])
    pwr_n_f = int(pwr_meta["meta"]["n_f"]) if pwr_meta is not None else int(n_f)
    feature_mode = str(load_meta["meta"].get("feature_mode", "base"))
    feature_runtime = _make_feature_runtime(load_meta["meta"], ts=float(ts))

    load_u_mode = str(load_meta["meta"].get("u_mode", "fixed"))
    pwr_u_mode = str(pwr_meta["meta"].get("u_mode", "fixed")) if pwr_meta is not None else None
    if pwr_meta is not None and str(pwr_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
        raise SystemExit("power model feature_mode mismatch")
    load_u_sched_ref = load_meta["meta"].get("u_sched_ref", None)
    pwr_u_sched_ref = pwr_meta["meta"].get("u_sched_ref", None) if pwr_meta is not None else None
    speed_meta = None
    speed_thetas = None
    speed_u_mode = None
    speed_u_sched_ref = None
    if speed_bundle is not None:
        speed_meta, speed_thetas = speed_bundle
        if list(speed_meta["feature_names"]) != list(feature_names):
            raise SystemExit("speed model feature_names mismatch")
        if int(speed_meta["meta"]["n_f"]) != int(n_f):
            raise SystemExit("speed model n_f mismatch")
        if str(speed_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
            raise SystemExit("speed model feature_mode mismatch")
        speed_u_mode = str(speed_meta["meta"].get("u_mode", "fixed"))
        speed_u_sched_ref = speed_meta["meta"].get("u_sched_ref", None)

    load_Af, load_B_blocks = _build_Af_B_blocks(load_thetas, n_f=n_f, N=N, u_mode=load_u_mode)
    pwr_Af = None
    pwr_B_blocks = None
    if pwr_thetas is not None and pwr_u_mode is not None:
        pwr_Af, pwr_B_blocks = _build_Af_B_blocks(pwr_thetas, n_f=pwr_n_f, N=N, u_mode=pwr_u_mode)
    speed_Af = None
    speed_B_blocks = None
    if speed_thetas is not None and speed_u_mode is not None:
        speed_Af, speed_B_blocks = _build_Af_B_blocks(speed_thetas, n_f=n_f, N=N, u_mode=speed_u_mode)
    secondary_meta = None
    secondary_Af = None
    secondary_B_blocks = None
    secondary_u_mode = None
    secondary_u_sched_ref = None
    if secondary_target is not None:
        secondary_meta, secondary_thetas = _load_model(args.models_dir, wind_mps=wind_mps, target=str(secondary_target))
        if list(secondary_meta["feature_names"]) != list(feature_names):
            raise SystemExit("secondary model feature_names mismatch")
        if int(secondary_meta["meta"]["n_f"]) != int(n_f):
            raise SystemExit("secondary model n_f mismatch")
        if str(secondary_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
            raise SystemExit("secondary model feature_mode mismatch")
        secondary_u_mode = str(secondary_meta["meta"].get("u_mode", "fixed"))
        secondary_u_sched_ref = secondary_meta["meta"].get("u_sched_ref", None)
        secondary_Af, secondary_B_blocks = _build_Af_B_blocks(secondary_thetas, n_f=n_f, N=N, u_mode=secondary_u_mode)

    blend_models_dir = args.blend_models_dir.resolve() if args.blend_models_dir is not None else None
    blend_enabled = blend_models_dir is not None and str(args.blend_mode) != "none"
    blend_load_Af = None
    blend_load_B_blocks = None
    blend_load_u_mode = None
    blend_load_u_sched_ref = None
    blend_pwr_Af = None
    blend_pwr_B_blocks = None
    blend_pwr_u_mode = None
    blend_pwr_u_sched_ref = None
    blend_speed_Af = None
    blend_speed_B_blocks = None
    blend_speed_u_mode = None
    blend_speed_u_sched_ref = None
    blend_secondary_Af = None
    blend_secondary_B_blocks = None
    blend_secondary_u_mode = None
    blend_secondary_u_sched_ref = None
    if blend_enabled:
        blend_load_meta, blend_load_thetas = _load_model(blend_models_dir, wind_mps=wind_mps, target=load_target)
        blend_pwr_meta = None
        blend_pwr_thetas = None
        if power_enable:
            blend_pwr_meta, blend_pwr_thetas = _load_model(blend_models_dir, wind_mps=wind_mps, target="GenPwr")
        speed_blend_models_dir = args.speed_blend_models_dir.resolve() if args.speed_blend_models_dir is not None else blend_models_dir
        blend_speed_bundle = None
        if speed_objective_active:
            blend_speed_bundle = _maybe_load_model(speed_blend_models_dir, wind_mps=wind_mps, target=str(args.speed_target))
        if list(blend_load_meta["feature_names"]) != list(feature_names):
            raise SystemExit("blend model feature_names mismatch; expected same feature basis for smooth blending")
        if int(blend_load_meta["meta"]["n_f"]) != int(n_f):
            raise SystemExit("blend model n_f mismatch")
        if str(blend_load_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
            raise SystemExit("blend model feature_mode mismatch")
        if blend_pwr_meta is not None and str(blend_pwr_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
            raise SystemExit("blend power model feature_mode mismatch")
        blend_load_u_mode = str(blend_load_meta["meta"].get("u_mode", "fixed"))
        blend_pwr_u_mode = str(blend_pwr_meta["meta"].get("u_mode", "fixed")) if blend_pwr_meta is not None else None
        blend_load_u_sched_ref = blend_load_meta["meta"].get("u_sched_ref", None)
        blend_pwr_u_sched_ref = blend_pwr_meta["meta"].get("u_sched_ref", None) if blend_pwr_meta is not None else None
        blend_load_Af, blend_load_B_blocks = _build_Af_B_blocks(blend_load_thetas, n_f=n_f, N=N, u_mode=blend_load_u_mode)
        if blend_pwr_thetas is not None and blend_pwr_u_mode is not None:
            blend_pwr_n_f = int(blend_pwr_meta["meta"]["n_f"])
            blend_pwr_Af, blend_pwr_B_blocks = _build_Af_B_blocks(blend_pwr_thetas, n_f=blend_pwr_n_f, N=N, u_mode=blend_pwr_u_mode)
        if blend_speed_bundle is not None:
            blend_speed_meta, blend_speed_thetas = blend_speed_bundle
            if list(blend_speed_meta["feature_names"]) != list(feature_names):
                raise SystemExit("blend speed model feature_names mismatch")
            if int(blend_speed_meta["meta"]["n_f"]) != int(n_f):
                raise SystemExit("blend speed model n_f mismatch")
            blend_speed_u_mode = str(blend_speed_meta["meta"].get("u_mode", "fixed"))
            blend_speed_u_sched_ref = blend_speed_meta["meta"].get("u_sched_ref", None)
            blend_speed_Af, blend_speed_B_blocks = _build_Af_B_blocks(blend_speed_thetas, n_f=n_f, N=N, u_mode=blend_speed_u_mode)
        if secondary_target is not None:
            blend_secondary_meta, blend_secondary_thetas = _load_model(blend_models_dir, wind_mps=wind_mps, target=str(secondary_target))
            if list(blend_secondary_meta["feature_names"]) != list(feature_names):
                raise SystemExit("blend secondary model feature_names mismatch")
            if int(blend_secondary_meta["meta"]["n_f"]) != int(n_f):
                raise SystemExit("blend secondary model n_f mismatch")
            blend_secondary_u_mode = str(blend_secondary_meta["meta"].get("u_mode", "fixed"))
            blend_secondary_u_sched_ref = blend_secondary_meta["meta"].get("u_sched_ref", None)
            blend_secondary_Af, blend_secondary_B_blocks = _build_Af_B_blocks(blend_secondary_thetas, n_f=n_f, N=N, u_mode=blend_secondary_u_mode)

    # Use the base (unscheduled) block for QP setup; we can still apply scheduling in q updates.
    load_B0 = load_B_blocks[0]
    pwr_B0 = pwr_B_blocks[0] if pwr_B_blocks is not None else None
    speed_B0 = speed_B_blocks[0] if speed_B_blocks is not None else None

    # Normalize load proxy to reduce unit/scale dominance in the QP.
    # (Without this, load terms in kN-m can overwhelm u/du penalties by many orders of magnitude.)
    y_std = float(load_meta["meta"].get("y_std", 1.0))
    if not math.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    load_Af = load_Af / y_std
    load_B_blocks = [B / y_std for B in load_B_blocks]
    load_B0 = load_B_blocks[0]
    if secondary_Af is not None and secondary_B_blocks is not None and secondary_meta is not None:
        secondary_y_std = float(secondary_meta["meta"].get("y_std", 1.0))
        if not math.isfinite(secondary_y_std) or secondary_y_std <= 0:
            secondary_y_std = 1.0
        secondary_Af = secondary_Af / secondary_y_std
        secondary_B_blocks = [B / secondary_y_std for B in secondary_B_blocks]
    if blend_enabled and blend_load_Af is not None and blend_load_B_blocks is not None:
        blend_y_std = float(blend_load_meta["meta"].get("y_std", 1.0))
        if not math.isfinite(blend_y_std) or blend_y_std <= 0:
            blend_y_std = 1.0
        blend_load_Af = blend_load_Af / blend_y_std
        blend_load_B_blocks = [B / blend_y_std for B in blend_load_B_blocks]
    # Keep speed-guard predictors in physical units.
    # Unlike load proxies, speed enters as a hard/soft physical guard rather than an objective term,
    # so applying y_std normalization here would silently mis-scale the constraint RHS.
    if blend_enabled and blend_secondary_Af is not None and blend_secondary_B_blocks is not None and secondary_target is not None:
        blend_secondary_y_std = float(blend_secondary_meta["meta"].get("y_std", 1.0))
        if not math.isfinite(blend_secondary_y_std) or blend_secondary_y_std <= 0:
            blend_secondary_y_std = 1.0
        blend_secondary_Af = blend_secondary_Af / blend_secondary_y_std
        blend_secondary_B_blocks = [B / blend_secondary_y_std for B in blend_secondary_B_blocks]

    # QP constants.
    u_max = math.radians(float(args.u_max_deg))
    du_max = math.radians(float(args.du_max_deg_s)) * float(ts)
    eq_lowfreq = [float(x.strip()) for x in args.eq_lowfreq_hz.split(",") if x.strip()]
    u_freq_centers = [float(x.strip()) for x in str(args.u_freq_centers_hz).split(",") if x.strip()]
    w_du_chain = float(args.w_du) if not math.isfinite(float(args.w_du_chain)) else float(args.w_du_chain)
    w_du_anchor = float(args.w_du) if not math.isfinite(float(args.w_du_anchor)) else float(args.w_du_anchor)
    w_load_schedule_table = _parse_schedule_triplets(args.w_load_schedule_rbf_table)
    blend_schedule_table = _parse_schedule_triplets(args.blend_rbf_table)
    synth_b_gamma_schedule_table = _parse_schedule_triplets(args.synth_b_gamma_rbf_table)
    primary_q_shrink_schedule_table = _parse_schedule_triplets(args.primary_q_shrink_rbf_table)
    primary_a_blend_schedule_table = _parse_schedule_triplets(args.primary_a_blend_rbf_table)
    tower_gate_op_schedule_table = _parse_schedule_triplets(args.tower_gate_op_rbf_table)
    act_limit_scale_schedule_table = _parse_schedule_triplets(args.act_limit_scale_rbf_table)
    tower_guard_mu_grid = [float(x) for x in _parse_floats(str(args.tower_guard_mus))] if str(args.tower_guard_mus).strip() else []
    tower_guard_mu_grid = sorted(set(float(x) for x in tower_guard_mu_grid if math.isfinite(float(x)) and float(x) >= 0.0))
    if not tower_guard_mu_grid:
        tower_guard_mu_grid = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    tower_primary_cap_rel = max(float(args.tower_primary_cap_rel), 0.0)
    tower_primary_cap_floor_rel = max(float(args.tower_primary_cap_floor_rel), 0.0)

    # Reference 1P frequency for multi-band load penalty.
    f_1p_ref_hz = float(args.load_1p_ref_hz)
    if not (math.isfinite(f_1p_ref_hz) and f_1p_ref_hz > 0):
        omega_ref = None
        if isinstance(load_u_sched_ref, dict) and "omega_ref" in load_u_sched_ref:
            omega_ref = float(load_u_sched_ref.get("omega_ref", float("nan")))
        elif isinstance(pwr_u_sched_ref, dict) and "omega_ref" in pwr_u_sched_ref:
            omega_ref = float(pwr_u_sched_ref.get("omega_ref", float("nan")))
        if omega_ref is not None and math.isfinite(omega_ref) and omega_ref > 0:
            f_1p_ref_hz = omega_ref / (2.0 * math.pi)
        else:
            f_1p_ref_hz = float("nan")

    qp = _build_qp(
        N=N,
        ts=ts,
        move_blocking_spec=str(args.move_blocking_spec),
        u_max=u_max,
        du_max=du_max,
        w_u=float(args.w_u),
        w_du=float(args.w_du),
        w_du_chain=float(w_du_chain),
        w_du_anchor=float(w_du_anchor),
        decision_coords=str(args.decision_coords),
        w_u_freq=float(args.w_u_freq),
        u_freq_centers_hz=u_freq_centers,
        u_freq_halfwidth_hz=float(args.u_freq_halfwidth_hz),
        u_freq_taps=int(args.u_freq_taps),
        load_B0=load_B0,
        load_Af=load_Af,
        load_B_blocks=load_B_blocks,
        load_u_mode=load_u_mode,
        load_u_sched_ref=load_u_sched_ref,
        bp_center_hz=float(args.bp_center_hz),
        bp_halfwidth_hz=float(args.bp_halfwidth_hz),
        bp_taps=int(args.bp_taps),
        w_load=float(args.w_load),
        w_load_mode=str(args.w_load_mode),
        w_load_auto_min=float(args.w_load_auto_min),
        w_load_auto_max=float(args.w_load_auto_max),
        load_rel_1p=float(args.load_1p_rel),
        load_rel_3p=float(args.load_3p_rel),
        load_1p_halfwidth_hz=float(args.load_1p_halfwidth_hz),
        load_3p_halfwidth_hz=float(args.load_3p_halfwidth_hz),
        load_1p_taps=int(args.load_1p_taps),
        load_3p_taps=int(args.load_3p_taps),
        load_freq_mode=str(args.load_freq_mode),
        load_q_mode=str(args.load_q_mode),
        tower_metric_mode=str(args.tower_metric_mode),
        tower_fatigue_alpha0=float(args.tower_fatigue_alpha0),
        tower_fatigue_alpha2=float(args.tower_fatigue_alpha2),
        tower_fatigue_alpha4=float(args.tower_fatigue_alpha4),
        tower_metric_normalize=str(args.tower_metric_normalize),
        tower_cone_alpha_band=float(args.tower_cone_alpha_band),
        tower_cone_alpha_m2=float(args.tower_cone_alpha_m2),
        tower_cone_alpha_m4=float(args.tower_cone_alpha_m4),
        tower_cone_alpha_env70=float(args.tower_cone_alpha_env70),
        tower_cone_alpha_env85=float(args.tower_cone_alpha_env85),
        tower_cone_alpha_env95=float(args.tower_cone_alpha_env95),
        tower_cone_alpha_envlp85=float(args.tower_cone_alpha_envlp85),
        tower_cone_alpha_envlp95=float(args.tower_cone_alpha_envlp95),
        tower_cone_alpha_tp90=float(args.tower_cone_alpha_tp90),
        tower_cone_alpha_tp95=float(args.tower_cone_alpha_tp95),
        tower_cone_alpha_sb1p=float(args.tower_cone_alpha_sb1p),
        tower_cone_alpha_sb3p=float(args.tower_cone_alpha_sb3p),
        tower_cone_alpha_sb1p_env95=float(args.tower_cone_alpha_sb1p_env95),
        tower_cone_alpha_sb3p_env95=float(args.tower_cone_alpha_sb3p_env95),
        tower_cone_env_floor=float(args.tower_cone_env_floor),
        tower_cone_env_sharpness=float(args.tower_cone_env_sharpness),
        tower_cone_lowfreq_cutoff_hz=float(args.tower_cone_lowfreq_cutoff_hz),
        tower_cone_tp_window_s=float(args.tower_cone_tp_window_s),
        tower_cone_basis_normalize=str(args.tower_cone_basis_normalize),
        f_1p_ref_hz=float(f_1p_ref_hz),
        eq_dc=bool(args.eq_dc),
        eq_lowfreq_hz=eq_lowfreq,
        power_enable=power_enable,
        power_eps=float(args.power_eps),
        power_B0=pwr_B0,
        power_Af=pwr_Af,
        power_B_blocks=pwr_B_blocks,
        power_u_mode=pwr_u_mode,
        power_u_sched_ref=pwr_u_sched_ref,
        speed_enable=bool(args.speed_enable),
        speed_form=str(args.speed_form),
        speed_upper=float(args.speed_upper),
        speed_target=str(args.speed_target),
        speed_soft_enable=bool(args.speed_soft_enable),
        speed_soft_weight=float(args.speed_soft_weight),
        speed_B0=speed_B0,
        speed_Af=speed_Af,
        speed_B_blocks=speed_B_blocks,
        speed_u_mode=speed_u_mode,
        speed_u_sched_ref=speed_u_sched_ref,
        secondary_cap_enable=bool(secondary_target is not None and secondary_cap_rel > 0.0),
        primary_cap_enable=bool(tower_primary_cap_rel > 0.0),
    )
    if speed_q_shrink_active and speed_Af is not None and speed_B_blocks is not None and speed_u_mode is not None:
        qp.Af_speed = speed_Af
        qp.B_blocks_speed = speed_B_blocks
        qp.u_mode_speed = speed_u_mode
        qp.u_sched_ref_speed = speed_u_sched_ref
        if qp.speed_target is None:
            qp.speed_target = str(args.speed_target)
        if qp.speed_form is None:
            qp.speed_form = str(args.primary_q_speed_form)
        if qp.speed_upper is None or not math.isfinite(float(qp.speed_upper)):
            speed_upper_q = float(args.primary_q_speed_upper)
            if not math.isfinite(speed_upper_q):
                speed_upper_q = float(args.speed_upper)
            qp.speed_upper = float(speed_upper_q)
    print(
        f"[zmq-mpc] objective={args.objective} ws={wind_mps} N={N} ts={ts:g} "
        f"move_blocks={len(qp.move_blocks)} move_block_spec={str(args.move_blocking_spec) or 'none'} "
        f"decision_coords={str(args.decision_coords)} "
        f"u_mode={load_u_mode} w_load_eff={qp.w_load:g} (mode={args.w_load_mode}, w_load_arg={args.w_load:g}) "
        f"load_rel_1p={float(args.load_1p_rel):g} load_rel_3p={float(args.load_3p_rel):g} "
        f"tower_metric_mode={args.tower_metric_mode} tower_metric_normalize={args.tower_metric_normalize} "
        f"load_freq_mode={args.load_freq_mode} f1p_ref_hz={f_1p_ref_hz:g} "
        f"w_load_schedule={args.w_load_schedule_mode} blend_mode={args.blend_mode} "
        f"blend_structure={args.blend_structure} g_norm_mode={args.g_norm_mode} g_norm_form={args.g_norm_form} primary_form={args.primary_form} "
        f"synth_b_mode={args.synth_b_mode} synth_b_apply={args.synth_b_apply} synth_b_source={args.synth_b_source} "
        f"synth_b_energy={float(args.synth_b_energy_retain):g} synth_b_gamma_mode={args.synth_b_gamma_mode} "
        f"primary_q_shrink_mode={args.primary_q_shrink_mode} "
        f"primary_q_speed_form={args.primary_q_speed_form} "
        f"primary_q_speed_upper={float(args.primary_q_speed_upper) if math.isfinite(float(args.primary_q_speed_upper)) else float(args.speed_upper):g} "
        f"primary_q_speed_weight={float(args.primary_q_speed_weight):g} primary_q_speed_grid={int(args.primary_q_speed_grid)} "
        f"primary_q_value_speed_weight={float(args.primary_q_value_speed_weight):g} "
        f"primary_q_value_trust_weight={float(args.primary_q_value_trust_weight):g} "
        f"speed_enable={bool(args.speed_enable)} speed_target={args.speed_target} speed_form={args.speed_form} "
        f"speed_soft_enable={bool(args.speed_soft_enable)} speed_soft_weight={float(args.speed_soft_weight):g} "
        f"speed_row_normalize={bool(args.speed_row_normalize)} "
        f"secondary_objective={args.secondary_objective} secondary_rel={secondary_rel:g} secondary_form={args.secondary_form} "
        f"secondary_cap_rel={secondary_cap_rel:g} "
        f"tower_structured_form={args.tower_structured_form} tower_structured_shape_gain={float(args.tower_structured_shape_gain):g} "
        f"tower_guard_eps={float(args.tower_guard_eps):g} tower_guard_mus={','.join(f'{x:g}' for x in tower_guard_mu_grid)} "
        f"tower_gate_env_center={float(args.tower_gate_env_center):g} tower_gate_env_width={float(args.tower_gate_env_width):g} "
        f"tower_gate_op_mode={args.tower_gate_op_mode} act_limit_scale_mode={args.act_limit_scale_mode} "
        f"tower_primary_cap_rel={tower_primary_cap_rel:g} tower_irls_iters={int(args.tower_irls_iters)}"
    )
    if bool(args.dry_run):
        return 0

    # Logging.
    csv_fh = None
    csv_wr = None
    if args.log_csv is not None:
        import csv

        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = args.log_csv.open("w", newline="", encoding="utf-8")
        csv_wr = csv.writer(csv_fh)
        csv_wr.writerow(
            [
                "msg_idx",
                "len",
                "t_s",
                "u_prev_rad",
                "u_cmd_rad",
                "u_cmd_deg",
                "qp_status",
                "qp_time_ms",
                "sched_wind_f_mps",
                "sched_beta_f_deg",
                "act_limit_scale",
                "blend_gamma",
                "synth_b_gamma",
                "synth_b_rank",
                "synth_b_rank_frac",
                "synth_b_energy_frac",
                "primary_q_lambda",
                "primary_q_lambda_ref",
                "primary_q_speed_penalty",
                "primary_q_value_rel",
                "primary_q_trust_penalty",
                "primary_q_kkt_active_count",
                "primary_q_kkt_active_rank",
                "primary_q_kkt_active_cond",
                "primary_a_eta",
                "primary_q_a_gap_norm",
                "primary_q_a_gap_rel",
                "blend_structure_is_g_only",
                "g_norm_scale",
                "g_norm_metric_ref",
                "g_norm_metric_eff",
                "w_load_eff_step",
                "tower_metric_pafs_z0_rms",
                "tower_metric_pafs_z0_inf",
                "tower_metric_pafs_w_mean",
                "tower_metric_pafs_w_p95",
                "tower_metric_pafs_w_top10_share",
                "tower_metric_hybrid_env_trace_frac_raw",
                "tower_metric_hybrid_spec_trace_frac_raw",
                "tower_metric_psd_band_trace_frac_raw",
                "tower_metric_psd_m2_trace_frac_raw",
                "tower_metric_psd_m4_trace_frac_raw",
                "tower_metric_psd_env_trace_frac_raw",
                "tower_metric_psd_sideband_trace_frac_raw",
                "tower_metric_psd_shape_trace_frac_raw",
                "tower_metric_psd_env_severity",
                "tower_structured_gate_chi",
                "tower_structured_gate_env",
                "tower_structured_gate_op",
                "tower_structured_guard_mu",
                "tower_structured_guard_band_ref",
                "tower_structured_guard_band_sel",
                "tower_structured_guard_shape_sel",
                "tower_structured_guard_feasible",
                "tower_primary_cap_floor",
                "tower_primary_cap_free_rms",
                "tower_primary_cap_active_frac",
                "tower_primary_cap_peak_ratio",
                "tower_irls_active",
                "tower_irls_weight_p95",
                "tower_irls_weight_top10_share",
                "speed_guard_upper",
                "speed_guard_form",
                "speed_guard_target",
                "speed_guard_rhs_min",
                "speed_guard_row_norm_mean",
                "speed_guard_row_norm_max",
                "speed_soft_slack",
            ]
            + meas_names
        )
        csv_fh.flush()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)

    u_prev = 0.0
    u_prev_prev = 0.0
    u_prev_ref_state = 0.0
    n = 0
    t_wall0 = time.time()
    sched_wind_f = float(wind_mps)
    sched_beta_f_deg = 0.0
    try:
        while True:
            msg = sock.recv_string()
            vals = _parse_floats(msg)
            n += 1
            # Map measurements by name; if message short, fill NaNs.
            meas = {name: float("nan") for name in meas_names}
            for name, v in zip(meas_names, vals):
                meas[name] = float(v)

            t_s = float(meas.get("Time", float("nan")))
            wind_now = _safe_float(meas.get("HorWindV", wind_mps), float(wind_mps))
            beta_now_deg = math.degrees(_safe_float(meas.get("BlPitchCMeas", 0.0), 0.0))
            if n == 1:
                sched_wind_f = wind_now
                sched_beta_f_deg = beta_now_deg
            else:
                sched_wind_f = _lp1(sched_wind_f, wind_now, ts=ts, tau_s=float(args.schedule_tau_s))
                sched_beta_f_deg = _lp1(sched_beta_f_deg, beta_now_deg, ts=ts, tau_s=float(args.schedule_tau_s))

            act_limit_scale = _evaluate_schedule(
                mode=str(args.act_limit_scale_mode),
                wind_f_mps=float(sched_wind_f),
                beta_f_deg=float(sched_beta_f_deg),
                sigmoid_center=float(args.act_limit_scale_center),
                sigmoid_width=float(args.act_limit_scale_width),
                sigmoid_lo=float(args.act_limit_scale_lo),
                sigmoid_hi=float(args.act_limit_scale_hi),
                rbf_table=act_limit_scale_schedule_table,
                rbf_v_sigma=float(args.act_limit_scale_rbf_v_sigma),
                rbf_beta_sigma_deg=float(args.act_limit_scale_rbf_beta_sigma_deg),
                default_value=1.0,
                log_interp=False,
            )
            act_limit_scale = float(np.clip(act_limit_scale, 1e-3, 1.0))
            u_max_runtime = float(u_max) * float(act_limit_scale)
            du_max_runtime = float(du_max) * float(act_limit_scale)

            if str(args.u_prev_ref_mode) == "avg2":
                u_prev_ref = 0.5 * (float(u_prev) + float(u_prev_prev))
            elif str(args.u_prev_ref_mode) == "ema":
                alpha_u = float(np.clip(float(args.u_prev_ref_alpha), 0.0, 1.0))
                if n == 1:
                    u_prev_ref_state = float(u_prev)
                else:
                    u_prev_ref_state = (1.0 - alpha_u) * float(u_prev_ref_state) + alpha_u * float(u_prev)
                u_prev_ref = float(u_prev_ref_state)
            else:
                u_prev_ref = float(u_prev)
            if abs(float(u_prev_ref)) < float(args.u_prev_ref_deadband_rad):
                u_prev_ref = 0.0

            if feature_runtime is not None:
                try:
                    meas_feature = feature_runtime.update(meas, u_prev=float(u_prev))
                except TypeError:
                    meas_feature = feature_runtime.update(meas)
            else:
                meas_feature = meas
            f_k = _feature_vector(meas_feature, feature_names)
            if pwr_Af is not None and list(pwr_feature_names) == list(feature_names):
                f_k_pwr = f_k
            elif pwr_Af is not None:
                f_k_pwr = _feature_vector(meas_feature, pwr_feature_names)
            else:
                f_k_pwr = None
            a_load_primary = qp.Af_load @ f_k
            B_eff_primary = _effective_B(
                blocks=qp.B_blocks_load,
                u_mode=qp.u_mode_load,
                sched_ref=qp.u_sched_ref_load,
                meas=meas,
                wind_mps=wind_mps,
            )
            a_speed_primary = None
            B_eff_speed_primary = None
            if qp.Af_speed is not None and qp.B_blocks_speed is not None and qp.u_mode_speed is not None:
                a_speed_primary = qp.Af_speed @ f_k
                B_eff_speed_primary = _effective_B(
                    blocks=qp.B_blocks_speed,
                    u_mode=qp.u_mode_speed,
                    sched_ref=qp.u_sched_ref_speed,
                    meas=meas,
                    wind_mps=wind_mps,
                )

            gamma_blend = 0.0
            a_load_blend = None
            if blend_enabled and blend_load_Af is not None and blend_load_B_blocks is not None and blend_load_u_mode is not None:
                gamma_blend = _evaluate_schedule(
                    mode=str(args.blend_mode),
                    wind_f_mps=float(sched_wind_f),
                    beta_f_deg=float(sched_beta_f_deg),
                    sigmoid_center=float(args.blend_center),
                    sigmoid_width=float(args.blend_width),
                    sigmoid_lo=float(args.blend_gamma_lo),
                    sigmoid_hi=float(args.blend_gamma_hi),
                    rbf_table=blend_schedule_table,
                    rbf_v_sigma=float(args.blend_rbf_v_sigma),
                    rbf_beta_sigma_deg=float(args.blend_rbf_beta_sigma_deg),
                    default_value=0.0,
                    log_interp=False,
                )
                gamma_blend = float(np.clip(gamma_blend, 0.0, 1.0))
                a_load_blend = blend_load_Af @ f_k
                B_eff_blend = _effective_B(
                    blocks=blend_load_B_blocks,
                    u_mode=blend_load_u_mode,
                    sched_ref=blend_load_u_sched_ref,
                    meas=meas,
                    wind_mps=wind_mps,
                )
                if str(args.blend_structure) == "g_only_load":
                    a_load = a_load_primary
                else:
                    a_load = (1.0 - gamma_blend) * a_load_primary + gamma_blend * a_load_blend
                B_eff = (1.0 - gamma_blend) * B_eff_primary + gamma_blend * B_eff_blend
            else:
                a_load = a_load_primary
                B_eff = B_eff_primary
            a_speed = a_speed_primary
            B_eff_speed = B_eff_speed_primary
            if (
                blend_enabled
                and blend_speed_Af is not None
                and blend_speed_B_blocks is not None
                and blend_speed_u_mode is not None
                and a_speed_primary is not None
                and B_eff_speed_primary is not None
            ):
                a_speed_blend = blend_speed_Af @ f_k
                B_eff_speed_blend = _effective_B(
                    blocks=blend_speed_B_blocks,
                    u_mode=blend_speed_u_mode,
                    sched_ref=blend_speed_u_sched_ref,
                    meas=meas,
                    wind_mps=wind_mps,
                )
                a_speed = (1.0 - gamma_blend) * a_speed_primary + gamma_blend * a_speed_blend
                B_eff_speed = (1.0 - gamma_blend) * B_eff_speed_primary + gamma_blend * B_eff_speed_blend
            a_load_secondary = None
            a_load_secondary_primary = None
            B_eff_secondary = None
            B_eff_secondary_primary = None
            B_qp_secondary = None
            if secondary_Af is not None and secondary_B_blocks is not None and secondary_u_mode is not None and (secondary_rel > 0.0 or secondary_cap_rel > 0.0):
                a_load_secondary_primary = secondary_Af @ f_k
                B_eff_secondary_primary = _effective_B(
                    blocks=secondary_B_blocks,
                    u_mode=secondary_u_mode,
                    sched_ref=secondary_u_sched_ref,
                    meas=meas,
                    wind_mps=wind_mps,
                )
                if (
                    blend_enabled
                    and blend_secondary_Af is not None
                    and blend_secondary_B_blocks is not None
                    and blend_secondary_u_mode is not None
                ):
                    a_load_secondary_blend = blend_secondary_Af @ f_k
                    B_eff_secondary_blend = _effective_B(
                        blocks=blend_secondary_B_blocks,
                        u_mode=blend_secondary_u_mode,
                        sched_ref=blend_secondary_u_sched_ref,
                        meas=meas,
                        wind_mps=wind_mps,
                    )
                    if str(args.blend_structure) == "g_only_load":
                        a_load_secondary = a_load_secondary_primary
                    else:
                        a_load_secondary = (1.0 - gamma_blend) * a_load_secondary_primary + gamma_blend * a_load_secondary_blend
                    B_eff_secondary = (1.0 - gamma_blend) * B_eff_secondary_primary + gamma_blend * B_eff_secondary_blend
                else:
                    a_load_secondary = a_load_secondary_primary
                    B_eff_secondary = B_eff_secondary_primary

            if str(qp.decision_coords) == "delta_u":
                u_hold_vec = float(u_prev) * qp.ones_u
                C_u = qp.C_u
                a_load_primary = a_load_primary + B_eff_primary @ u_hold_vec
                B_eff_primary = B_eff_primary @ C_u
                if a_load_blend is not None and blend_enabled:
                    a_load_blend = a_load_blend + B_eff_blend @ u_hold_vec
                    B_eff_blend = B_eff_blend @ C_u
                a_load = a_load + B_eff @ u_hold_vec
                B_eff = B_eff @ C_u
                if a_speed_primary is not None and B_eff_speed_primary is not None:
                    a_speed_primary = a_speed_primary + B_eff_speed_primary @ u_hold_vec
                    B_eff_speed_primary = B_eff_speed_primary @ C_u
                if a_speed is not None and B_eff_speed is not None:
                    a_speed = a_speed + B_eff_speed @ u_hold_vec
                    B_eff_speed = B_eff_speed @ C_u
                if a_load_secondary_primary is not None and B_eff_secondary_primary is not None:
                    a_load_secondary_primary = a_load_secondary_primary + B_eff_secondary_primary @ u_hold_vec
                    B_eff_secondary_primary = B_eff_secondary_primary @ C_u
                if a_load_secondary is not None and B_eff_secondary is not None:
                    a_load_secondary = a_load_secondary + B_eff_secondary @ u_hold_vec
                    B_eff_secondary = B_eff_secondary @ C_u

            pafs_stats = {
                "pafs_z0_rms": float("nan"),
                "pafs_z0_inf": float("nan"),
                "pafs_w_mean": float("nan"),
                "pafs_w_p95": float("nan"),
                "pafs_w_top10_share": float("nan"),
                "hybrid_env_trace_frac_raw": float("nan"),
                "hybrid_spec_trace_frac_raw": float("nan"),
                "psd_band_trace_frac_raw": float("nan"),
                "psd_m2_trace_frac_raw": float("nan"),
                "psd_m4_trace_frac_raw": float("nan"),
                "psd_env_trace_frac_raw": float("nan"),
                "psd_sideband_trace_frac_raw": float("nan"),
            }
            tower_structured_chi = float("nan")
            tower_structured_env_gate = float("nan")
            tower_structured_op_gate = float("nan")
            tower_structured_guard_mu = float("nan")
            tower_structured_guard_band_ref = float("nan")
            tower_structured_guard_band_sel = float("nan")
            tower_structured_guard_shape_sel = float("nan")
            tower_structured_guard_feasible = 0.0
            M_band_step = qp.M_tower
            M_shape_step = np.zeros_like(qp.M_tower)
            M_tower_step = qp.M_tower
            if str(args.tower_metric_mode) == "pafs_online":
                M_tower_step, pafs_stats = _build_pafs_metric(
                    H_tower=qp.H_tower,
                    a_load=a_load,
                    alpha=float(args.tower_pafs_alpha),
                    m_w=float(args.tower_pafs_mw),
                    eps=float(args.tower_pafs_eps),
                    normalize_mode=str(args.tower_metric_normalize),
                )
            elif str(args.tower_metric_mode) == "fatigue_hybrid_env":
                M_tower_step, pafs_stats = _build_hybrid_fatigue_metric(
                    H_tower=qp.H_tower,
                    a_load=a_load,
                    ts=float(qp.ts),
                    alpha0=float(args.tower_fatigue_alpha0),
                    alpha2=float(args.tower_fatigue_alpha2),
                    alpha4=float(args.tower_fatigue_alpha4),
                    alphae=float(args.tower_fatigue_alphae),
                    pafs_alpha=float(args.tower_pafs_alpha),
                    pafs_mw=float(args.tower_pafs_mw),
                    pafs_eps=float(args.tower_pafs_eps),
                    normalize_mode=str(args.tower_metric_normalize),
                )
            elif str(args.tower_metric_mode) == "psd_cone":
                M_tower_step, pafs_stats = _build_psd_cone_metric(
                    H_tower=qp.H_tower,
                    a_load=a_load,
                    ts=float(qp.ts),
                    bp_center_hz=float(args.bp_center_hz),
                    bp_halfwidth_hz=float(args.bp_halfwidth_hz),
                    bp_taps=int(args.bp_taps),
                    f_1p_ref_hz=float(qp.f_1p_ref_hz),
                    alpha_band=float(args.tower_cone_alpha_band),
                    alpha_m2=float(args.tower_cone_alpha_m2),
                    alpha_m4=float(args.tower_cone_alpha_m4),
                    alpha_env70=float(args.tower_cone_alpha_env70),
                    alpha_env85=float(args.tower_cone_alpha_env85),
                    alpha_env95=float(args.tower_cone_alpha_env95),
                    alpha_envlp85=float(args.tower_cone_alpha_envlp85),
                    alpha_envlp95=float(args.tower_cone_alpha_envlp95),
                    alpha_tp90=float(args.tower_cone_alpha_tp90),
                    alpha_tp95=float(args.tower_cone_alpha_tp95),
                    alpha_sb1p=float(args.tower_cone_alpha_sb1p),
                    alpha_sb3p=float(args.tower_cone_alpha_sb3p),
                    alpha_sb1p_env95=float(args.tower_cone_alpha_sb1p_env95),
                    alpha_sb3p_env95=float(args.tower_cone_alpha_sb3p_env95),
                    env_floor=float(args.tower_cone_env_floor),
                    env_sharpness=float(args.tower_cone_env_sharpness),
                    lowfreq_cutoff_hz=float(args.tower_cone_lowfreq_cutoff_hz),
                    tp_window_s=float(args.tower_cone_tp_window_s),
                    basis_normalize=str(args.tower_cone_basis_normalize),
                    normalize_mode=str(args.tower_metric_normalize),
                )
                if str(args.tower_structured_form) in ("guarded_shape", "state_gate"):
                    M_band_step, M_shape_step, M_psd_full, psd_parts_stats = _build_psd_cone_components(
                        H_tower=qp.H_tower,
                        a_load=a_load,
                        ts=float(qp.ts),
                        bp_center_hz=float(args.bp_center_hz),
                        bp_halfwidth_hz=float(args.bp_halfwidth_hz),
                        bp_taps=int(args.bp_taps),
                        f_1p_ref_hz=float(qp.f_1p_ref_hz),
                        alpha_band=float(args.tower_cone_alpha_band),
                        alpha_m2=float(args.tower_cone_alpha_m2),
                        alpha_m4=float(args.tower_cone_alpha_m4),
                        alpha_env70=float(args.tower_cone_alpha_env70),
                        alpha_env85=float(args.tower_cone_alpha_env85),
                        alpha_env95=float(args.tower_cone_alpha_env95),
                        alpha_envlp85=float(args.tower_cone_alpha_envlp85),
                        alpha_envlp95=float(args.tower_cone_alpha_envlp95),
                        alpha_tp90=float(args.tower_cone_alpha_tp90),
                        alpha_tp95=float(args.tower_cone_alpha_tp95),
                        alpha_sb1p=float(args.tower_cone_alpha_sb1p),
                        alpha_sb3p=float(args.tower_cone_alpha_sb3p),
                        alpha_sb1p_env95=float(args.tower_cone_alpha_sb1p_env95),
                        alpha_sb3p_env95=float(args.tower_cone_alpha_sb3p_env95),
                        env_floor=float(args.tower_cone_env_floor),
                        env_sharpness=float(args.tower_cone_env_sharpness),
                        lowfreq_cutoff_hz=float(args.tower_cone_lowfreq_cutoff_hz),
                        tp_window_s=float(args.tower_cone_tp_window_s),
                        basis_normalize=str(args.tower_cone_basis_normalize),
                        normalize_mode=str(args.tower_metric_normalize),
                    )
                    pafs_stats.update(psd_parts_stats)
                    if str(args.tower_structured_form) == "state_gate":
                        tower_structured_chi, tower_structured_env_gate, tower_structured_op_gate = _structured_gate_value(
                            severity=float(psd_parts_stats.get("psd_env_severity", float("nan"))),
                            env_center=float(args.tower_gate_env_center),
                            env_width=float(args.tower_gate_env_width),
                            op_mode=str(args.tower_gate_op_mode),
                            wind_f_mps=float(sched_wind_f),
                            beta_f_deg=float(sched_beta_f_deg),
                            op_center=float(args.tower_gate_op_center),
                            op_width=float(args.tower_gate_op_width),
                            op_lo=float(args.tower_gate_op_lo),
                            op_hi=float(args.tower_gate_op_hi),
                            op_rbf_table=tower_gate_op_schedule_table,
                            op_rbf_v_sigma=float(args.tower_gate_op_rbf_v_sigma),
                            op_rbf_beta_sigma_deg=float(args.tower_gate_op_rbf_beta_sigma_deg),
                        )
                        M_tower_step = M_band_step + float(args.tower_structured_shape_gain) * float(tower_structured_chi) * M_shape_step
                    else:
                        M_tower_step = M_psd_full

            tower_primary_cap_floor = float("nan")
            tower_primary_cap_free_rms = float("nan")
            tower_primary_cap_active_frac = float("nan")
            tower_primary_cap_peak_ratio = float("nan")
            tower_irls_active = 0.0
            tower_irls_weight_p95 = float("nan")
            tower_irls_weight_top10_share = float("nan")

            w_load_step = _evaluate_schedule(
                mode=str(args.w_load_schedule_mode),
                wind_f_mps=float(sched_wind_f),
                beta_f_deg=float(sched_beta_f_deg),
                sigmoid_center=float(args.w_load_schedule_center),
                sigmoid_width=float(args.w_load_schedule_width),
                sigmoid_lo=float(args.w_load_schedule_lo),
                sigmoid_hi=float(args.w_load_schedule_hi),
                rbf_table=w_load_schedule_table,
                rbf_v_sigma=float(args.w_load_schedule_rbf_v_sigma),
                rbf_beta_sigma_deg=float(args.w_load_schedule_rbf_beta_sigma_deg),
                default_value=float(qp.w_load),
                log_interp=True,
            )
            w_load_step = float(max(w_load_step, 0.0))
            primary_q_mode = str(args.primary_q_shrink_mode)
            if str(args.primary_form) == "full":
                primary_q_lambda = 1.0
            elif str(args.primary_form) == "hessian_only":
                primary_q_lambda = 0.0
            elif str(args.primary_form) == "q_only_a_blend":
                primary_q_lambda = float("nan")
            elif primary_q_mode in ("local_opt_gamma", "local_opt_gamma_kkt", "local_opt_speed", "local_opt_speed_project", "local_opt_value_speed"):
                primary_q_lambda = float("nan")
            else:
                primary_q_lambda = _evaluate_schedule(
                    mode=primary_q_mode,
                    wind_f_mps=float(sched_wind_f),
                    beta_f_deg=float(sched_beta_f_deg),
                    sigmoid_center=float(args.primary_q_shrink_center),
                    sigmoid_width=float(args.primary_q_shrink_width),
                    sigmoid_lo=float(args.primary_q_shrink_lo),
                    sigmoid_hi=float(args.primary_q_shrink_hi),
                    rbf_table=primary_q_shrink_schedule_table,
                    rbf_v_sigma=float(args.primary_q_shrink_rbf_v_sigma),
                    rbf_beta_sigma_deg=float(args.primary_q_shrink_rbf_beta_sigma_deg),
                    default_value=1.0,
                    log_interp=False,
                )
                primary_q_lambda = float(np.clip(primary_q_lambda, 0.0, 1.0))

            # For LPV (scheduled) input sensitivity, the correct QP Hessian depends on B_eff:
            #   J_load = w (a + B_eff u)^T M_total (a + B_eff u)
            #   P_load = 2 w B_eff^T M_total B_eff
            #
            # The initial setup used the unscheduled block (B0) to define a fixed sparsity pattern.
            # Updating Px here keeps the problem convex and significantly reduces schedule mismatch.
            rot_speed_radps = float(meas.get("RotSpeedF", meas.get("RotSpeed", float("nan"))))
            M_total_static, _ = _get_M_total_and_maybe_update_cache(qp, rot_speed_radps=rot_speed_radps)
            if str(args.tower_metric_mode) in ("pafs_online", "fatigue_hybrid_env", "psd_cone"):
                M_total = M_total_static + (M_tower_step - qp.M_tower)
            else:
                M_total = M_total_static
            B_qp, g_norm_scale, g_norm_metric_ref, g_norm_metric_eff = _normalized_B_for_qp(
                B_ref=B_eff_primary,
                B_eff=B_eff,
                M_total=M_total,
                mode=str(args.g_norm_mode),
                form=str(args.g_norm_form),
                scale_min=float(args.g_norm_min),
                scale_max=float(args.g_norm_max),
                eps=float(args.g_norm_eps),
            )
            B_qp_primary, _, _, _ = _normalized_B_for_qp(
                B_ref=B_eff_primary,
                B_eff=B_eff_primary,
                M_total=M_total,
                mode=str(args.g_norm_mode),
                form=str(args.g_norm_form),
                scale_min=float(args.g_norm_min),
                scale_max=float(args.g_norm_max),
                eps=float(args.g_norm_eps),
            )
            synth_b_gamma = 0.0
            synth_b_rank = float("nan")
            synth_b_rank_frac = float("nan")
            synth_b_energy_frac = float("nan")
            B_qp_load_hess_primary = B_qp
            B_qp_load_q_primary = B_qp
            if str(args.synth_b_mode) == "svd_energy":
                synth_b_gamma = _evaluate_schedule(
                    mode=str(args.synth_b_gamma_mode),
                    wind_f_mps=float(sched_wind_f),
                    beta_f_deg=float(sched_beta_f_deg),
                    sigmoid_center=float(args.synth_b_gamma_center),
                    sigmoid_width=float(args.synth_b_gamma_width),
                    sigmoid_lo=float(args.synth_b_gamma_lo),
                    sigmoid_hi=float(args.synth_b_gamma_hi),
                    rbf_table=synth_b_gamma_schedule_table,
                    rbf_v_sigma=float(args.synth_b_gamma_rbf_v_sigma),
                    rbf_beta_sigma_deg=float(args.synth_b_gamma_rbf_beta_sigma_deg),
                    default_value=0.0,
                    log_interp=False,
                )
                synth_b_gamma = float(np.clip(synth_b_gamma, float(args.synth_b_gamma_lo), float(args.synth_b_gamma_hi)))
                B_synth_source = B_qp_primary if str(args.synth_b_source) == "primary" else B_qp
                B_lowrank, rank_keep, energy_frac = _svd_energy_lowrank(
                    B_synth_source,
                    energy_retain=float(args.synth_b_energy_retain),
                    min_rank=max(int(args.synth_b_min_rank), 1),
                    max_rank=max(int(args.synth_b_max_rank), 0),
                )
                synth_b_rank = float(rank_keep)
                synth_b_rank_frac = float(rank_keep / max(1, min(B_synth_source.shape)))
                synth_b_energy_frac = float(energy_frac)
                B_qp_synth = (1.0 - synth_b_gamma) * B_qp + synth_b_gamma * B_lowrank
                B_qp_load_hess_primary = B_qp_synth
                if str(args.synth_b_apply) == "hessian_and_q":
                    B_qp_load_q_primary = B_qp_synth
            g_norm_metric_scale = float(g_norm_scale * g_norm_scale)
            g_norm_metric_scale_primary = 1.0
            if str(args.g_norm_form) == "metric_only":
                P_load_u_primary = (2.0 * w_load_step * g_norm_metric_scale) * (B_qp_load_hess_primary.T @ M_total @ B_qp_load_hess_primary)
            else:
                P_load_u_primary = (2.0 * w_load_step) * (B_qp_load_hess_primary.T @ M_total @ B_qp_load_hess_primary)
            P_load_u_secondary = np.zeros_like(P_load_u_primary)
            if a_load_secondary is not None and B_eff_secondary is not None and B_eff_secondary_primary is not None and secondary_rel > 0.0:
                B_qp_secondary, g_norm_scale_secondary, _, _ = _normalized_B_for_qp(
                    B_ref=B_eff_secondary_primary,
                    B_eff=B_eff_secondary,
                    M_total=M_total,
                    mode=str(args.g_norm_mode),
                    form=str(args.g_norm_form),
                    scale_min=float(args.g_norm_min),
                    scale_max=float(args.g_norm_max),
                    eps=float(args.g_norm_eps),
                )
                if str(args.secondary_form) in ("full", "hessian_only"):
                    if str(args.g_norm_form) == "metric_only":
                        sec_metric_scale = float(g_norm_scale_secondary * g_norm_scale_secondary)
                        P_load_u_secondary = (2.0 * w_load_step * float(secondary_rel) * sec_metric_scale) * (B_qp_secondary.T @ M_total @ B_qp_secondary)
                    else:
                        P_load_u_secondary = (2.0 * w_load_step * float(secondary_rel)) * (B_qp_secondary.T @ M_total @ B_qp_secondary)
            P_load_u = P_load_u_primary + P_load_u_secondary
            P_new = qp.P_const.copy()
            P_new[: qp.n_u, : qp.n_u] += P_load_u
            a_pwr = qp.Af_pwr @ f_k_pwr if (qp.idx_power is not None and qp.Af_pwr is not None) else None
            if a_pwr is not None and blend_enabled and blend_pwr_Af is not None and str(args.blend_structure) == "whole_model":
                a_pwr_blend = blend_pwr_Af @ f_k_pwr
                a_pwr = (1.0 - gamma_blend) * a_pwr + gamma_blend * a_pwr_blend

            M_q = M_total if str(qp.load_q_mode) == "full" else qp.M_tower
            tmp = M_q @ a_load
            tmp_q_primary = tmp
            primary_a_eta = float("nan")
            primary_q_a_gap_norm = float("nan")
            primary_q_a_gap_rel = float("nan")
            if str(args.primary_form) == "q_only_a_blend":
                if a_load_blend is not None:
                    primary_a_mode = str(args.primary_a_blend_mode)
                    if primary_a_mode in ("", "none", "match_gamma"):
                        primary_a_eta = float(gamma_blend)
                    elif primary_a_mode == "scale_gamma":
                        primary_a_eta = float(args.primary_a_blend_scale) * float(gamma_blend)
                    else:
                        primary_a_eta = _evaluate_schedule(
                            mode=primary_a_mode,
                            wind_f_mps=float(sched_wind_f),
                            beta_f_deg=float(sched_beta_f_deg),
                            sigmoid_center=float(args.primary_a_blend_center),
                            sigmoid_width=float(args.primary_a_blend_width),
                            sigmoid_lo=float(args.primary_a_blend_lo),
                            sigmoid_hi=float(args.primary_a_blend_hi),
                            rbf_table=primary_a_blend_schedule_table,
                            rbf_v_sigma=float(args.primary_a_blend_rbf_v_sigma),
                            rbf_beta_sigma_deg=float(args.primary_a_blend_rbf_beta_sigma_deg),
                            default_value=float(gamma_blend),
                            log_interp=False,
                        )
                    primary_a_eta = float(np.clip(primary_a_eta, float(args.primary_a_blend_lo), float(args.primary_a_blend_hi)))
                    a_q_primary = a_load_primary + primary_a_eta * (a_load_blend - a_load_primary)
                else:
                    primary_a_eta = 0.0
                    a_q_primary = a_load_primary
                tmp_q_primary = M_q @ a_q_primary
            primary_q_lambda_ref = float("nan")
            primary_q_speed_penalty = float("nan")
            primary_q_value_rel = float("nan")
            primary_q_trust_penalty = float("nan")
            primary_q_kkt_active_count = float("nan")
            primary_q_kkt_active_rank = float("nan")
            primary_q_kkt_active_cond = float("nan")
            if str(args.primary_form) == "q_shrink" and primary_q_mode in ("local_opt_gamma", "local_opt_gamma_kkt", "local_opt_speed", "local_opt_speed_project", "local_opt_value_speed"):
                dB_q = B_qp - B_qp_primary
                if str(args.g_norm_form) == "metric_only":
                    P_eff_local = g_norm_metric_scale * (B_qp.T @ M_total @ B_qp)
                    P_primary_local = g_norm_metric_scale_primary * (B_qp_primary.T @ M_total @ B_qp_primary)
                    dP_dgamma_u = (2.0 * w_load_step) * (P_eff_local - P_primary_local)
                else:
                    dP_dgamma_u = (2.0 * w_load_step) * (dB_q.T @ M_total @ B_qp + B_qp.T @ M_total @ dB_q)
                dq_primary_u = (2.0 * w_load_step) * (dB_q.T @ tmp)
                q_base_u = (2.0 * w_load_step) * (B_qp_primary.T @ tmp)
                q_primary_full = (2.0 * w_load_step) * (B_qp.T @ tmp)
                q_nom_u = q_primary_full.copy()
                if (
                    a_load_secondary is not None
                    and B_eff_secondary is not None
                    and B_eff_secondary_primary is not None
                    and secondary_rel > 0.0
                    and str(args.secondary_form) in ("full", "q_only")
                ):
                    tmp_secondary = M_q @ a_load_secondary
                    q_nom_u += (2.0 * w_load_step * float(secondary_rel)) * (B_qp_secondary.T @ tmp_secondary)
                    q_base_u += (2.0 * w_load_step * float(secondary_rel)) * (B_qp_secondary.T @ tmp_secondary)
                if qp.w_du_anchor > 0:
                    d0 = np.zeros(qp.n_u, dtype=float)
                    d0[0] = float(u_prev_ref)
                    q_nom_u += (-2.0 * qp.w_du_anchor) * qp.anchor_vec * float(d0[0])
                    q_base_u += (-2.0 * qp.w_du_anchor) * qp.anchor_vec * float(d0[0])
                q_nom = _embed_u_vector(q_nom_u, n_var=qp.n_var)
                dq_primary = _embed_u_vector(dq_primary_u, n_var=qp.n_var)
                dP_dgamma = _embed_u_matrix(dP_dgamma_u, n_var=qp.n_var)
                if primary_q_mode == "local_opt_gamma_kkt":
                    l_nom, u_nom_bounds, A_rebuild_dense_nom, _ = _build_runtime_constraint_system(
                        qp=qp,
                        args=args,
                        meas=meas,
                        u_prev=float(u_prev),
                        u_max_runtime=float(u_max),
                        du_max_runtime=float(du_max),
                        a_pwr=a_pwr,
                        a_speed=a_speed if (qp.idx_speed is not None or qp.idx_speed_soft is not None) else None,
                        B_eff_speed=B_eff_speed if (qp.idx_speed is not None or qp.idx_speed_soft is not None) else None,
                        a_load_primary=a_load,
                        B_qp_primary=B_qp,
                        primary_cap_rel=float(tower_primary_cap_rel),
                        primary_cap_floor_rel=float(tower_primary_cap_floor_rel),
                        a_load_secondary=a_load_secondary,
                        B_qp_secondary=B_qp_secondary,
                        secondary_cap_rel=float(secondary_cap_rel),
                    )
                    A_dense_nom = qp.A_dense_base if A_rebuild_dense_nom is None else A_rebuild_dense_nom
                    try:
                        P_sp_nom = sp.csc_matrix((_csc_triu_data(P_new), qp.P.indices, qp.P.indptr), shape=qp.P.shape)
                        A_sp_nom = sp.csc_matrix(A_dense_nom)
                        prob_nom = osqp.OSQP()
                        prob_nom.setup(P=P_sp_nom, q=q_nom, A=A_sp_nom, l=l_nom, u=u_nom_bounds, verbose=False, polish=False, warm_start=False)
                        res_nom = prob_nom.solve()
                        if res_nom.x is not None and "solved" in str(res_nom.info.status).lower():
                            x_nom = np.asarray(res_nom.x, dtype=float)
                            A_act = _active_rows_from_bounds(A_dense_nom, x_nom, l_nom, u_nom_bounds, tol=1e-7)
                            du_q, primary_q_kkt_active_cond, primary_q_kkt_active_rank = _solve_kkt_sensitivity(P_new, A_act, dq_primary)
                            du_P, _, _ = _solve_kkt_sensitivity(P_new, A_act, dP_dgamma @ x_nom)
                            primary_q_kkt_active_count = float(A_act.shape[0])
                            denom = float(np.dot(du_q, du_q))
                            if denom > 1e-18 and np.all(np.isfinite(du_q)) and np.all(np.isfinite(du_P)):
                                primary_q_lambda = float(-np.dot(du_P, du_q) / denom)
                            else:
                                primary_q_lambda = 1.0
                        else:
                            raise np.linalg.LinAlgError(f"nominal osqp status={getattr(res_nom.info, 'status', 'unknown')}")
                    except Exception:
                        try:
                            u_nom = -np.linalg.solve(P_new, q_nom)
                            du_q = -np.linalg.solve(P_new, dq_primary)
                            du_P = -np.linalg.solve(P_new, dP_dgamma @ u_nom)
                            denom = float(np.dot(du_q, du_q))
                            if denom > 1e-18:
                                primary_q_lambda = float(-np.dot(du_P, du_q) / denom)
                            else:
                                primary_q_lambda = 1.0
                        except np.linalg.LinAlgError:
                            primary_q_lambda = 1.0
                else:
                    try:
                        u_nom = -np.linalg.solve(P_new, q_nom)
                        du_q = -np.linalg.solve(P_new, dq_primary)
                        du_P = -np.linalg.solve(P_new, dP_dgamma @ u_nom)
                        denom = float(np.dot(du_q, du_q))
                        if denom > 1e-18:
                            primary_q_lambda = float(-np.dot(du_P, du_q) / denom)
                        else:
                            primary_q_lambda = 1.0
                    except np.linalg.LinAlgError:
                        primary_q_lambda = 1.0
                primary_q_lambda = float(np.clip(primary_q_lambda, float(args.primary_q_shrink_lo), float(args.primary_q_shrink_hi)))
                primary_q_lambda_ref = float(primary_q_lambda)
                if primary_q_mode in ("local_opt_speed", "local_opt_speed_project", "local_opt_value_speed"):
                    speed_upper_consistency = float(args.primary_q_speed_upper)
                    if not math.isfinite(speed_upper_consistency):
                        speed_upper_consistency = float(args.speed_upper)
                    if (
                        math.isfinite(speed_upper_consistency)
                        and a_speed is not None
                        and B_eff_speed is not None
                        and int(args.primary_q_speed_grid) >= 2
                    ):
                        q_base = _embed_u_vector(q_base_u, n_var=qp.n_var)
                        try:
                            u_base = -np.linalg.solve(P_new, q_base)
                            du_lambda = -np.linalg.solve(P_new, dq_primary)
                            u_base_u = u_base[: qp.n_u]
                            du_lambda_u = du_lambda[: qp.n_u]
                            speed_affine = a_speed + B_eff_speed @ u_base_u
                            speed_dir = B_eff_speed @ du_lambda_u
                            lam_lo = float(args.primary_q_shrink_lo)
                            lam_hi = float(args.primary_q_shrink_hi)
                            speed_scale = max(abs(speed_upper_consistency), 1e-6)
                            def _speed_penalty_for_lambda(lam: float) -> float:
                                speed_pred = speed_affine + float(lam) * speed_dir
                                if str(args.primary_q_speed_form) == "mean_upper":
                                    viol = max(float(np.mean(speed_pred) - speed_upper_consistency), 0.0) / speed_scale
                                    return float(viol * viol)
                                viol = np.maximum(speed_pred - speed_upper_consistency, 0.0) / speed_scale
                                return float(np.mean(viol * viol))

                            if primary_q_mode == "local_opt_speed_project":
                                lam_feas_lo = float(lam_lo)
                                lam_feas_hi = float(lam_hi)
                                form = str(args.primary_q_speed_form)
                                if form == "mean_upper":
                                    c = float(np.mean(speed_dir))
                                    b = float(np.mean(speed_affine) - speed_upper_consistency)
                                    if abs(c) <= 1e-12:
                                        if b > 0.0:
                                            lam_feas_lo = lam_hi + 1.0
                                            lam_feas_hi = lam_lo - 1.0
                                    elif c > 0.0:
                                        lam_feas_hi = min(lam_feas_hi, float(-b / c))
                                    else:
                                        lam_feas_lo = max(lam_feas_lo, float(-b / c))
                                else:
                                    for b_i, c_i in zip(speed_affine - speed_upper_consistency, speed_dir):
                                        b_i = float(b_i)
                                        c_i = float(c_i)
                                        if abs(c_i) <= 1e-12:
                                            if b_i > 0.0:
                                                lam_feas_lo = lam_hi + 1.0
                                                lam_feas_hi = lam_lo - 1.0
                                                break
                                            continue
                                        bound = float(-b_i / c_i)
                                        if c_i > 0.0:
                                            lam_feas_hi = min(lam_feas_hi, bound)
                                        else:
                                            lam_feas_lo = max(lam_feas_lo, bound)
                                if lam_feas_lo <= lam_feas_hi:
                                    primary_q_lambda = float(np.clip(primary_q_lambda_ref, lam_feas_lo, lam_feas_hi))
                                else:
                                    grid = np.linspace(lam_lo, lam_hi, int(args.primary_q_speed_grid), dtype=float)
                                    best_lambda = float(primary_q_lambda_ref)
                                    best_penalty = float("inf")
                                    for lam in grid:
                                        speed_pen = _speed_penalty_for_lambda(float(lam))
                                        if speed_pen < best_penalty:
                                            best_penalty = float(speed_pen)
                                            best_lambda = float(lam)
                                    primary_q_lambda = float(np.clip(best_lambda, lam_lo, lam_hi))
                                primary_q_speed_penalty = _speed_penalty_for_lambda(float(primary_q_lambda))
                            elif primary_q_mode == "local_opt_value_speed":
                                grid = np.linspace(lam_lo, lam_hi, int(args.primary_q_speed_grid), dtype=float)
                                grid = np.unique(np.concatenate([grid, np.array([primary_q_lambda_ref], dtype=float)]))
                                q_ref = q_base + float(primary_q_lambda_ref) * dq_primary
                                u_ref = -np.linalg.solve(P_new, q_ref)
                                v_ref = float(0.5 * u_ref.T @ P_new @ u_ref + q_ref.T @ u_ref)
                                value_scale = max(abs(v_ref), 1e-9)
                                speed_weight = max(float(args.primary_q_value_speed_weight), 0.0)
                                trust_weight = max(float(args.primary_q_value_trust_weight), 0.0)
                                trust_scale = max(lam_hi - lam_lo, 1e-9)
                                best_score = float("inf")
                                best_lambda = float(primary_q_lambda_ref)
                                best_speed_pen = float("nan")
                                best_value_rel = float("nan")
                                best_trust_pen = float("nan")
                                for lam in grid:
                                    q_l = q_base + float(lam) * dq_primary
                                    u_l = -np.linalg.solve(P_new, q_l)
                                    v_l = float(0.5 * u_l.T @ P_new @ u_l + q_l.T @ u_l)
                                    value_rel = float((v_l - v_ref) / value_scale)
                                    speed_pen = _speed_penalty_for_lambda(float(lam))
                                    trust_pen = float(((float(lam) - float(primary_q_lambda_ref)) / trust_scale) ** 2)
                                    score = float(value_rel + speed_weight * speed_pen + trust_weight * trust_pen)
                                    if score < best_score:
                                        best_score = score
                                        best_lambda = float(lam)
                                        best_speed_pen = float(speed_pen)
                                        best_value_rel = float(value_rel)
                                        best_trust_pen = float(trust_pen)
                                primary_q_lambda = float(np.clip(best_lambda, lam_lo, lam_hi))
                                primary_q_speed_penalty = float(best_speed_pen)
                                primary_q_value_rel = float(best_value_rel)
                                primary_q_trust_penalty = float(best_trust_pen)
                            else:
                                grid = np.linspace(lam_lo, lam_hi, int(args.primary_q_speed_grid), dtype=float)
                                grid = np.unique(np.concatenate([grid, np.array([primary_q_lambda_ref], dtype=float)]))
                                speed_weight = max(float(args.primary_q_speed_weight), 0.0)
                                best_score = float("inf")
                                best_lambda = float(primary_q_lambda_ref)
                                best_penalty = float("nan")
                                for lam in grid:
                                    speed_pen = _speed_penalty_for_lambda(float(lam))
                                    score = float((float(lam) - primary_q_lambda_ref) ** 2 + speed_weight * speed_pen)
                                    if score < best_score:
                                        best_score = score
                                        best_lambda = float(lam)
                                        best_penalty = float(speed_pen)
                                primary_q_lambda = float(np.clip(best_lambda, lam_lo, lam_hi))
                                primary_q_speed_penalty = float(best_penalty)
                        except np.linalg.LinAlgError:
                            primary_q_lambda = float(primary_q_lambda_ref)
            if str(args.primary_form) == "q_shrink":
                B_q_for_primary = (1.0 - primary_q_lambda) * B_qp_primary + primary_q_lambda * B_qp
            elif str(args.primary_form) == "q_only_a_blend":
                B_q_for_primary = B_qp_load_q_primary
            else:
                B_q_for_primary = B_qp_load_q_primary if str(args.primary_form) == "full" else B_qp_primary
            q_u_primary = (2.0 * w_load_step) * (B_q_for_primary.T @ tmp_q_primary)
            q_u = q_u_primary.copy()
            if str(args.primary_form) == "q_only_a_blend":
                q_a_base_u = (2.0 * w_load_step) * (B_qp.T @ (M_q @ a_load_primary))
                q_a_gap_u = q_u - q_a_base_u
                primary_q_a_gap_norm = float(np.linalg.norm(q_a_gap_u))
                primary_q_a_gap_rel = float(primary_q_a_gap_norm / max(float(np.linalg.norm(q_u)), 1e-18))
            if (
                a_load_secondary is not None
                and B_eff_secondary is not None
                and B_eff_secondary_primary is not None
                and secondary_rel > 0.0
                and str(args.secondary_form) in ("full", "q_only")
            ):
                tmp_secondary = M_q @ a_load_secondary
                q_u_secondary = (2.0 * w_load_step * float(secondary_rel)) * (B_qp_secondary.T @ tmp_secondary)
                q_u += q_u_secondary
            else:
                q_u_secondary = np.zeros_like(q_u_primary)

            q_u_abs = np.zeros_like(q_u_primary)
            if str(qp.decision_coords) == "delta_u":
                q_u_abs = qp.C_u.T @ (qp.P_abs_u @ (float(u_prev) * qp.ones_u))
                q_u += q_u_abs

            # du linear term: -2 w_du D^T d0, with d0=[u_prev,0,...]
            q_u_anchor = np.zeros_like(q_u_primary)
            if str(qp.decision_coords) != "delta_u" and qp.w_du_anchor > 0:
                d0 = np.zeros(qp.n_u, dtype=float)
                d0[0] = float(u_prev_ref)
                q_u_anchor = (-2.0 * qp.w_du_anchor) * qp.anchor_vec * float(d0[0])
                q_u += q_u_anchor
            q = _embed_u_vector(q_u, n_var=qp.n_var)

            # Update constraint bounds.
            l, u, A_rebuild_dense, constraint_stats = _build_runtime_constraint_system(
                qp=qp,
                args=args,
                meas=meas,
                u_prev=float(u_prev),
                u_max_runtime=float(u_max_runtime),
                du_max_runtime=float(du_max_runtime),
                a_pwr=a_pwr,
                a_speed=a_speed if (qp.idx_speed is not None or qp.idx_speed_soft is not None) else None,
                B_eff_speed=B_eff_speed if (qp.idx_speed is not None or qp.idx_speed_soft is not None) else None,
                a_load_primary=a_load,
                B_qp_primary=B_qp,
                primary_cap_rel=float(tower_primary_cap_rel),
                primary_cap_floor_rel=float(tower_primary_cap_floor_rel),
                a_load_secondary=a_load_secondary,
                B_qp_secondary=B_qp_secondary,
                secondary_cap_rel=float(secondary_cap_rel),
            )
            speed_guard_rhs_min = float(constraint_stats.get("speed_guard_rhs_min", float("nan")))
            speed_guard_row_norm_mean = float(constraint_stats.get("speed_guard_row_norm_mean", float("nan")))
            speed_guard_row_norm_max = float(constraint_stats.get("speed_guard_row_norm_max", float("nan")))
            tower_primary_cap_floor = float(constraint_stats.get("tower_primary_cap_floor", float("nan")))
            tower_primary_cap_free_rms = float(constraint_stats.get("tower_primary_cap_free_rms", float("nan")))
            speed_soft_slack = float("nan")

            structured_guard_active = (
                str(args.tower_metric_mode) == "psd_cone"
                and str(args.tower_structured_form) == "guarded_shape"
                and float(args.tower_structured_shape_gain) > 0.0
                and np.any(M_shape_step)
            )
            A_dense_solve = qp.A_dense_base if A_rebuild_dense is None else A_rebuild_dense

            def _solve_runtime_qp(P_mat: np.ndarray, q_vec: np.ndarray) -> osqp.OSQP_results:
                if A_rebuild_dense is not None:
                    P_sp_local = sp.csc_matrix((_csc_triu_data(P_mat), qp.P.indices, qp.P.indptr), shape=qp.P.shape)
                    A_sp_local = sp.csc_matrix(A_dense_solve)
                    prob_local = osqp.OSQP()
                    prob_local.setup(P=P_sp_local, q=q_vec, A=A_sp_local, l=l, u=u, verbose=False, polish=False, warm_start=False)
                    return prob_local.solve()
                qp.prob.update(Px=_csc_triu_data(P_mat))
                qp.prob.update(q=q_vec, l=l, u=u)
                return qp.prob.solve()

            # Solve.
            t0 = time.perf_counter()
            B_qp_solved = B_qp
            if structured_guard_active:
                shape_gain = float(args.tower_structured_shape_gain)
                M_band_total = np.asarray(M_total_static, dtype=float)
                M_band_q = np.asarray(M_total_static if str(qp.load_q_mode) == "full" else M_band_step, dtype=float)
                M_shape_total = shape_gain * np.asarray(M_shape_step, dtype=float)
                M_shape_q = M_shape_total

                def _build_primary_terms(M_total_local: np.ndarray, M_q_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    if str(args.g_norm_form) == "metric_only":
                        P_primary_local = (2.0 * w_load_step * g_norm_metric_scale) * (B_qp.T @ M_total_local @ B_qp)
                    else:
                        P_primary_local = (2.0 * w_load_step) * (B_qp.T @ M_total_local @ B_qp)
                    q_primary_local = (2.0 * w_load_step) * (B_q_for_primary.T @ (M_q_local @ a_load))
                    return np.asarray(P_primary_local, dtype=float), np.asarray(q_primary_local, dtype=float)

                P_band_primary, q_band_primary = _build_primary_terms(M_band_total, M_band_q)
                P_band = qp.P_const.copy()
                P_band[: qp.n_u, : qp.n_u] += P_band_primary + P_load_u_secondary
                q_band_u = q_band_primary + q_u_secondary + q_u_abs + q_u_anchor
                q_band = _embed_u_vector(q_band_u, n_var=qp.n_var)
                res_band = _solve_runtime_qp(P_band, q_band)
                status_band = str(res_band.info.status) if res_band.info is not None else "unknown"
                x_band = np.asarray(res_band.x, dtype=float) if (res_band.x is not None and "solved" in status_band.lower()) else None
                if x_band is None:
                    res = res_band
                    P_new = P_band
                    q = q_band
                    status = status_band
                else:
                    u_band = x_band[: qp.n_u]
                    tower_structured_guard_band_ref = _quadratic_prediction_cost(a_load, B_qp, M_band_total, u_band)
                    best_payload: tuple[np.ndarray, np.ndarray, osqp.OSQP_results, float, float, float] | None = None
                    feasible_count = 0
                    for mu in tower_guard_mu_grid:
                        M_total_local = M_shape_total + float(mu) * M_band_total
                        M_q_local = M_shape_q + float(mu) * M_band_q
                        P_primary_local, q_primary_local = _build_primary_terms(M_total_local, M_q_local)
                        P_local = qp.P_const.copy()
                        P_local[: qp.n_u, : qp.n_u] += P_primary_local + P_load_u_secondary
                        q_local_u = q_primary_local + q_u_secondary + q_u_abs + q_u_anchor
                        q_local = _embed_u_vector(q_local_u, n_var=qp.n_var)
                        res_local = _solve_runtime_qp(P_local, q_local)
                        status_local = str(res_local.info.status) if res_local.info is not None else "unknown"
                        if res_local.x is None or "solved" not in status_local.lower():
                            continue
                        x_local = np.asarray(res_local.x, dtype=float)
                        u_local = x_local[: qp.n_u]
                        band_local = _quadratic_prediction_cost(a_load, B_qp, M_band_total, u_local)
                        if not math.isfinite(tower_structured_guard_band_ref):
                            continue
                        if band_local > (1.0 + float(args.tower_guard_eps)) * tower_structured_guard_band_ref:
                            continue
                        feasible_count += 1
                        shape_local = _quadratic_prediction_cost(a_load, B_qp, M_shape_total, u_local)
                        reg_local = float(0.5 * u_local.T @ qp.P_const[: qp.n_u, : qp.n_u] @ u_local + q_u_anchor.T @ u_local)
                        score_local = float(shape_local + reg_local)
                        if best_payload is None or score_local < best_payload[3]:
                            best_payload = (P_local, q_local, res_local, score_local, float(mu), band_local)
                            tower_structured_guard_shape_sel = float(shape_local)
                    tower_structured_guard_feasible = 1.0 if feasible_count > 0 else 0.0
                    if best_payload is not None:
                        P_new, q, res, _, tower_structured_guard_mu, tower_structured_guard_band_sel = best_payload
                        status = str(res.info.status) if res.info is not None else "unknown"
                    else:
                        P_new = P_band
                        q = q_band
                        res = res_band
                        status = status_band
                        tower_structured_guard_mu = -1.0
                        tower_structured_guard_band_sel = tower_structured_guard_band_ref
                        tower_structured_guard_shape_sel = _quadratic_prediction_cost(a_load, B_qp, M_shape_total, u_band)
            else:
                res = _solve_runtime_qp(P_new, q)
                status = str(res.info.status) if res.info is not None else "unknown"
                if (
                    int(args.tower_irls_iters) > 0
                    and str(args.primary_form) == "full"
                    and res.x is not None
                    and "solved" in status.lower()
                ):
                    tower_irls_active = 1.0
                    irls_iters = max(int(args.tower_irls_iters), 0)
                    irls_m = max(float(args.tower_irls_m), 2.0)
                    irls_eps = max(float(args.tower_irls_eps), 1e-12)
                    irls_clip = max(float(args.tower_irls_w_clip), 1.0)
                    irls_rel = float(args.tower_irls_band_rel)
                    irls_res = res
                    irls_P = P_new
                    irls_q = q
                    irls_B = B_qp
                    irls_g_norm_scale = g_norm_scale
                    irls_M_total = M_total
                    for _ in range(irls_iters):
                        u_irls = np.asarray(irls_res.x[: qp.n_u], dtype=float)
                        z_irls = qp.H_tower @ (a_load + irls_B @ u_irls)
                        w_irls = np.power(np.abs(z_irls) + irls_eps, irls_m - 2.0)
                        w_irls = w_irls / max(float(np.mean(w_irls)), irls_eps)
                        w_irls = np.clip(w_irls, 0.0, irls_clip)
                        w_irls = w_irls / max(float(np.mean(w_irls)), irls_eps)
                        tower_irls_weight_p95 = float(np.percentile(w_irls, 95.0))
                        topn = max(1, int(math.ceil(0.1 * w_irls.size)))
                        tower_irls_weight_top10_share = float(np.sort(w_irls)[-topn:].sum() / max(float(w_irls.sum()), irls_eps))
                        M_band_irls = qp.H_tower.T @ (w_irls[:, None] * qp.H_tower)
                        irls_M_total = M_total + float(irls_rel) * (M_band_irls - qp.M_tower)
                        M_q_irls = irls_M_total if str(qp.load_q_mode) == "full" else (qp.M_tower + float(irls_rel) * (M_band_irls - qp.M_tower))
                        irls_B, irls_g_norm_scale, _, _ = _normalized_B_for_qp(
                            B_ref=B_eff_primary,
                            B_eff=B_eff,
                            M_total=irls_M_total,
                            mode=str(args.g_norm_mode),
                            form=str(args.g_norm_form),
                            scale_min=float(args.g_norm_min),
                            scale_max=float(args.g_norm_max),
                            eps=float(args.g_norm_eps),
                        )
                        if str(args.g_norm_form) == "metric_only":
                            P_load_u_irls = (2.0 * w_load_step * float(irls_g_norm_scale * irls_g_norm_scale)) * (irls_B.T @ irls_M_total @ irls_B)
                        else:
                            P_load_u_irls = (2.0 * w_load_step) * (irls_B.T @ irls_M_total @ irls_B)
                        irls_P = qp.P_const.copy()
                        irls_P[: qp.n_u, : qp.n_u] += P_load_u_irls + P_load_u_secondary
                        tmp_irls = M_q_irls @ a_load
                        q_u_irls = (2.0 * w_load_step) * (irls_B.T @ tmp_irls) + q_u_secondary + q_u_abs + q_u_anchor
                        irls_q = _embed_u_vector(q_u_irls, n_var=qp.n_var)
                        irls_res = _solve_runtime_qp(irls_P, irls_q)
                        status = str(irls_res.info.status) if irls_res.info is not None else "unknown"
                        if irls_res.x is None or "solved" not in status.lower():
                            break
                    if irls_res.x is not None and "solved" in status.lower():
                        res = irls_res
                        P_new = irls_P
                        q = irls_q
                        B_qp_solved = irls_B
                        g_norm_scale = float(irls_g_norm_scale)
            t_ms = (time.perf_counter() - t0) * 1000.0

            u_cmd = 0.0
            if res.x is not None and "solved" in status.lower():
                if str(qp.decision_coords) == "delta_u":
                    v_sol = np.asarray(res.x[: qp.n_u], dtype=float)
                    u_abs = float(u_prev) * qp.ones_u + qp.C_u @ v_sol
                    u_cmd = float(u_abs[0])
                else:
                    u_cmd = float(res.x[0])
                if qp.idx_speed_slack is not None and int(qp.idx_speed_slack) < len(res.x):
                    speed_soft_slack = float(res.x[int(qp.idx_speed_slack)])
            else:
                u_cmd = 0.0

            # Clamp for safety.
            u_cmd = float(np.clip(u_cmd, -u_max_runtime, u_max_runtime))

            if (
                tower_primary_cap_rel > 0.0
                and res.x is not None
                and "solved" in status.lower()
                and math.isfinite(tower_primary_cap_floor)
            ):
                u_sol = np.asarray(res.x[: qp.n_u], dtype=float)
                z_pred = qp.H_tower @ (a_load + B_qp_solved @ u_sol)
                z_free = qp.H_tower @ a_load
                cap_vec = np.maximum(float(tower_primary_cap_floor), float(tower_primary_cap_rel) * np.abs(z_free))
                tower_primary_cap_active_frac = float(np.mean(np.abs(z_pred) >= 0.98 * cap_vec))
                tower_primary_cap_peak_ratio = float(np.max(np.abs(z_pred) / np.maximum(cap_vec, 1e-9)))

            if csv_wr is not None:
                def _metric_stat(name: str) -> float:
                    return float(pafs_stats.get(name, float("nan")))

                row = [
                    n,
                    len(vals),
                    t_s,
                    u_prev,
                    u_cmd,
                    math.degrees(u_cmd),
                    status,
                    t_ms,
                    float(sched_wind_f),
                    float(sched_beta_f_deg),
                    float(act_limit_scale),
                    float(gamma_blend),
                    float(synth_b_gamma),
                    float(synth_b_rank),
                    float(synth_b_rank_frac),
                    float(synth_b_energy_frac),
                    float(primary_q_lambda),
                    float(primary_q_lambda_ref),
                    float(primary_q_speed_penalty),
                    float(primary_q_value_rel),
                    float(primary_q_trust_penalty),
                    float(primary_q_kkt_active_count),
                    float(primary_q_kkt_active_rank),
                    float(primary_q_kkt_active_cond),
                    float(primary_a_eta),
                    float(primary_q_a_gap_norm),
                    float(primary_q_a_gap_rel),
                    1.0 if str(args.blend_structure) == "g_only_load" else 0.0,
                    float(g_norm_scale),
                    float(g_norm_metric_ref),
                    float(g_norm_metric_eff),
                    float(w_load_step),
                    _metric_stat("pafs_z0_rms"),
                    _metric_stat("pafs_z0_inf"),
                    _metric_stat("pafs_w_mean"),
                    _metric_stat("pafs_w_p95"),
                    _metric_stat("pafs_w_top10_share"),
                    _metric_stat("hybrid_env_trace_frac_raw"),
                    _metric_stat("hybrid_spec_trace_frac_raw"),
                    _metric_stat("psd_band_trace_frac_raw"),
                    _metric_stat("psd_m2_trace_frac_raw"),
                    _metric_stat("psd_m4_trace_frac_raw"),
                    _metric_stat("psd_env_trace_frac_raw"),
                    _metric_stat("psd_sideband_trace_frac_raw"),
                    _metric_stat("psd_shape_trace_frac_raw"),
                    _metric_stat("psd_env_severity"),
                    float(tower_structured_chi),
                    float(tower_structured_env_gate),
                    float(tower_structured_op_gate),
                    float(tower_structured_guard_mu),
                    float(tower_structured_guard_band_ref),
                    float(tower_structured_guard_band_sel),
                    float(tower_structured_guard_shape_sel),
                    float(tower_structured_guard_feasible),
                    float(tower_primary_cap_floor),
                    float(tower_primary_cap_free_rms),
                    float(tower_primary_cap_active_frac),
                    float(tower_primary_cap_peak_ratio),
                    float(tower_irls_active),
                    float(tower_irls_weight_p95),
                    float(tower_irls_weight_top10_share),
                    float(qp.speed_upper) if qp.speed_upper is not None else float("nan"),
                    str(qp.speed_form) if qp.speed_form is not None else "",
                    str(qp.speed_target) if qp.speed_target is not None else "",
                    float(speed_guard_rhs_min),
                    float(speed_guard_row_norm_mean),
                    float(speed_guard_row_norm_max),
                    float(speed_soft_slack),
                ]
                row.extend([meas[nm] for nm in meas_names])
                csv_wr.writerow(row)
                if (n % 50) == 0:
                    csv_fh.flush()

            if (n % max(int(args.log_every), 1)) == 0:
                dt = time.time() - t_wall0
                rate = n / dt if dt > 0 else float("nan")
                print(
                    f"[zmq-mpc] n={n} len={len(vals)} expected={expected_n} t={t_s:.3f} rate={rate:.1f} msg/s "
                    f"u_cmd_deg={math.degrees(u_cmd):+.3f} gamma={gamma_blend:.3f} eta={primary_a_eta:.3f} qlam={primary_q_lambda:.3f} "
                    f"qlam_ref={primary_q_lambda_ref:.3f} speed_pen={primary_q_speed_penalty:.3g} qval={primary_q_value_rel:.3g} qtrust={primary_q_trust_penalty:.3g} "
                    f"kkt_n={primary_q_kkt_active_count:.0f} "
                    f"g_norm={g_norm_scale:.3f} w_load={w_load_step:.3g} speed_slack={speed_soft_slack:.4g} "
                    f"status={status} solve_ms={t_ms:.2f}"
                )

            # Respond with 8 setpoints, keep defaults for non-pitch fields.
            out = list(idef.setpoints_default)
            out[2] = u_cmd
            out[3] = u_cmd
            out[4] = u_cmd
            reply = ", ".join(f"{v:016.8e}" for v in out)
            sock.send_string(reply)

            u_prev_prev = float(u_prev)
            u_prev = u_cmd
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
        if csv_fh is not None:
            csv_fh.flush()
            csv_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
