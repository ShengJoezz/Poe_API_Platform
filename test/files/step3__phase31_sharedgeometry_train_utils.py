from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
REPO = PROGRAM_ROOT.parents[1]
BASE_ROOT = REPO / "Exp" / "algebraic_cleanup_probe_2026-03-25"
PHASE16_ROOT = REPO / "Exp" / "control_geometry_step2_program_2026-03-28" / "code" / "phase16_actuator_state_poweraware_2026_03_28"
PHASE18_ROOT = REPO / "Exp" / "control_geometry_step2_program_2026-03-28" / "code" / "phase18_shared_kernel_blocksep_2026_03_28"
for p in (BASE_ROOT, PHASE16_ROOT, PHASE18_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.train_algebraic_cleanup_hybrid as base_train  # noqa: E402
from algebraic_cleanup_proxy import simulate_algebraic_cleanup_df  # noqa: E402
from actstate_train_utils import (  # noqa: E402
    _assemble_current_design,
    _assemble_paired_design,
    _ridge_solve,
    _u_sched_reference,
)
from scripts.stage4_zmq_train_direct_mstep import (  # noqa: E402
    RunData,
    _build_kernel_prior_penalty_matrix,
    _build_toeplitz_kernel_design,
    _expand_toeplitz_kernel,
    _solve_constrained_joint_ridge,
    u_blocks_for_mode,
)

from phase31_proxy import (  # noqa: E402
    TARGETS,
    TARGET_FIT_WEIGHTS,
    family_cfg,
    feature_names_for_family_target,
    horizon_weight,
    make_algebraic_cleanup_params,
)


U_MODE = "tv_omega_beta_wind2_centered"
GM_U_SOURCE = "actuator_delta"
GM_SCHEDULE_SOURCE = "reference"
RIDGE_A = 1e-2
RIDGE_G = 1e-2
TOWER_TARGETS = {"FA_Acc_TT", "TwrBsMyt"}


def augment_meas_df(meas_df: pd.DataFrame) -> pd.DataFrame:
    df = meas_df.copy()
    psi1 = np.deg2rad(df["Azimuth"].to_numpy(dtype=float))
    psi2 = psi1 + 2.0 * np.pi / 3.0
    psi3 = psi1 + 4.0 * np.pi / 3.0
    m1 = df["rootMOOP(1)"].to_numpy(dtype=float)
    m2 = df["rootMOOP(2)"].to_numpy(dtype=float)
    m3 = df["rootMOOP(3)"].to_numpy(dtype=float)
    root0 = (m1 + m2 + m3) / 3.0
    root1c = (2.0 / 3.0) * (m1 * np.cos(psi1) + m2 * np.cos(psi2) + m3 * np.cos(psi3))
    root1s = (2.0 / 3.0) * (m1 * np.sin(psi1) + m2 * np.sin(psi2) + m3 * np.sin(psi3))
    rs = df["RotSpeed"].to_numpy(dtype=float)
    df["RootM0"] = root0
    df["RootM1c"] = root1c
    df["RootM1s"] = root1s
    df["RootM1_amp"] = np.sqrt(root1c**2 + root1s**2)
    df["RootM0_ma8"] = pd.Series(root0).rolling(window=8, min_periods=1).mean().to_numpy(dtype=float)
    df["RootM0_trend16"] = pd.Series(root0).diff(16).fillna(0.0).to_numpy(dtype=float)
    df["RotSpeed_ma8"] = pd.Series(rs).rolling(window=8, min_periods=1).mean().to_numpy(dtype=float)
    df["RotSpeed_trend16"] = pd.Series(rs).diff(16).fillna(0.0).to_numpy(dtype=float)
    if "VS_GenPwr_W" in df.columns and "VS_GenPwr" not in df.columns:
        df["VS_GenPwr"] = df["VS_GenPwr_W"]
    return df


def collect_augmented_runs(
    root: Path,
    *,
    winds: set[int],
    seeds: set[int],
) -> Dict[tuple[int, int], base_train.RawRun]:
    raw_lookup = base_train._collect_runs(root, winds=winds, seeds=seeds)
    out: Dict[tuple[int, int], base_train.RawRun] = {}
    for key, raw in raw_lookup.items():
        meas_aug = raw.meas_df.copy()
        if not raw.of_df.empty:
            for col in ("TwrBsMyt", "FA_Acc_TT"):
                if col in raw.of_df.columns and col not in meas_aug.columns:
                    meas_aug[col] = raw.of_df[col].to_numpy(dtype=float)
        out[key] = replace(raw, meas_df=augment_meas_df(meas_aug))
    return out


def _prepare_run_data(
    raw: base_train.RawRun,
    *,
    refs: dict,
    params: dict,
    load_proj: dict,
    feature_names: Sequence[str],
) -> RunData:
    aug = simulate_algebraic_cleanup_df(
        raw.meas_df,
        refs=refs,
        params=params,
        load_proj=load_proj,
        ts=float(raw.ts),
    )
    for col in (
        "GenSpeed",
        "GenTqMeas",
        "VS_GenPwr",
        "VS_GenPwr_W",
        "RootM0",
        "RootM1c",
        "RootM1s",
        "RootM1_amp",
        "RootM0_ma8",
        "RootM0_trend16",
        "RotSpeed_ma8",
        "RotSpeed_trend16",
    ):
        if col in raw.meas_df.columns:
            aug[col] = raw.meas_df[col].to_numpy(dtype=float)
    f = aug.loc[:, list(feature_names)].to_numpy(dtype=float)
    y = {}
    for name in TARGETS:
        if name in raw.meas_df.columns:
            y[name] = raw.meas_df[name].to_numpy(dtype=float)
        elif name in raw.of_df.columns:
            y[name] = raw.of_df[name].to_numpy(dtype=float)
    return RunData(
        run_dir=raw.run_dir,
        wind_mps=int(raw.wind_mps),
        seed=int(raw.seed),
        ts=float(raw.ts),
        t=raw.meas_df["Time"].to_numpy(dtype=float),
        u=raw.meas_df["pit_offset_rad"].to_numpy(dtype=float),
        omega0=raw.meas_df["RotSpeed"].to_numpy(dtype=float),
        beta0=raw.meas_df["BlPitchCMeas"].to_numpy(dtype=float),
        wind0=raw.meas_df["HorWindV"].to_numpy(dtype=float),
        f=f,
        y=y,
    )


def build_runs_by_wind(
    *,
    raw_lookup: Dict[tuple[int, int], base_train.RawRun],
    winds: Sequence[int],
    seeds: Sequence[int],
    family: str,
    N: int,
    ts: float,
    t_start_s: float,
) -> tuple[Dict[int, Dict[str, List[RunData]]], Dict[int, dict]]:
    params = make_algebraic_cleanup_params()
    out: Dict[int, Dict[str, List[RunData]]] = {}
    runtime_meta_by_ws: Dict[int, dict] = {}
    for ws in winds:
        raws = [
            raw_lookup[(int(ws), int(sd))]
            for sd in seeds
            if (int(ws), int(sd)) in raw_lookup
            and base_train._run_has_min_samples(raw_lookup[(int(ws), int(sd))], N=N, t_start_s=t_start_s, ts=ts)
        ]
        if not raws:
            out[int(ws)] = {}
            continue
        refs, load_proj = base_train._build_runtime_for_ws(raws, params=params, t_start_s=t_start_s, ts=ts)
        runtime_meta_by_ws[int(ws)] = base_train._feature_runtime_meta(
            family=family,
            refs=refs,
            params=params,
            load_proj=load_proj,
        )
        out[int(ws)] = {}
        for target in TARGETS:
            feats = feature_names_for_family_target(family, target)
            out[int(ws)][target] = [
                _prepare_run_data(r, refs=refs, params=params, load_proj=load_proj, feature_names=feats)
                for r in raws
            ]
    return out, runtime_meta_by_ws


def _build_target_dataset(
    *,
    runs_ws: Sequence[RunData],
    ref_lookup_ws: Dict[tuple[int, int], RunData],
    target: str,
    N: int,
    ts: float,
    t_start_s: float,
    horizon_weight_decay: float,
) -> dict:
    XtX_blocks: List[np.ndarray] = []
    Xty_blocks: List[np.ndarray] = []
    current_designs: List[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    horizon_weights: List[float] = []
    y_std: float | None = None
    u_sched_ref = _u_sched_reference(
        runs=list(runs_ws),
        N=int(N),
        ts=float(ts),
        t_start_s=float(t_start_s),
        u_mode=U_MODE,
    )
    for p in range(1, int(N) + 1):
        wp = horizon_weight(int(p), decay=float(horizon_weight_decay))
        Xf_all, Xu_cur_all, yp_all = _assemble_current_design(
            runs=list(runs_ws),
            target=str(target),
            p=int(p),
            N=int(N),
            ts=float(ts),
            t_start_s=float(t_start_s),
            u_mode=U_MODE,
            u_sched_ref=u_sched_ref,
        )
        if Xf_all.size == 0 or Xu_cur_all.size == 0 or yp_all.size == 0:
            raise RuntimeError(f"no samples available for target={target} p={p}")
        if p == 1:
            s = float(np.std(yp_all))
            y_std = s if (math.isfinite(s) and s > 0.0) else 1.0
        Xu_pair_all, ypair_all = _assemble_paired_design(
            runs=list(runs_ws),
            ref_lookup=ref_lookup_ws,
            target=str(target),
            p=int(p),
            ts=float(ts),
            t_start_s=float(t_start_s),
            u_mode=U_MODE,
            u_sched_ref=u_sched_ref,
            gm_u_source=GM_U_SOURCE,
            gm_schedule_source=GM_SCHEDULE_SOURCE,
        )
        Zp_all = _build_toeplitz_kernel_design(u_feat=Xu_pair_all, p=int(p), N=int(N), u_mode=U_MODE)
        XtX_blocks.append(float(wp) * (Zp_all.T @ Zp_all))
        Xty_blocks.append(float(wp) * (Zp_all.T @ ypair_all))
        current_designs.append((Xf_all, Xu_cur_all, yp_all, Xu_pair_all, int(p)))
        horizon_weights.append(float(wp))
    if y_std is None:
        raise RuntimeError(f"failed to compute y_std for target={target}")
    fit_weight = float(TARGET_FIT_WEIGHTS[str(target)]) / max(float(y_std) ** 2, 1e-12)
    return {
        "target": str(target),
        "y_std": float(y_std),
        "n_f": int(current_designs[0][0].shape[1]),
        "current_designs": current_designs,
        "XtX_blocks": XtX_blocks,
        "Xty_blocks": Xty_blocks,
        "fit_weight": float(fit_weight),
        "horizon_weights": horizon_weights,
        "u_sched_ref": u_sched_ref,
    }


def _solve_indep_kernel(
    *,
    data: dict,
    kernel_prior_penalty: np.ndarray | None,
    ridge_alpha_g: float,
) -> np.ndarray:
    agg_XtX = float(data["fit_weight"]) * np.sum(np.stack(data["XtX_blocks"], axis=0), axis=0)
    agg_Xty = float(data["fit_weight"]) * np.sum(np.stack(data["Xty_blocks"], axis=0), axis=0)
    return _solve_constrained_joint_ridge(
        XtX_blocks=[agg_XtX],
        Xty_blocks=[agg_Xty],
        ridge_alpha=float(ridge_alpha_g),
        C=np.zeros((0, agg_XtX.shape[0]), dtype=float),
        d=np.zeros((0,), dtype=float),
        extra_penalty=kernel_prior_penalty,
    )


def _orth_components(Xf: np.ndarray, Xu: np.ndarray, *, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    XtX = Xu.T @ Xu + float(eps) * np.eye(Xu.shape[1], dtype=float)
    proj = np.linalg.solve(XtX, Xu.T @ Xf)
    Xf_proj = Xu @ proj
    return Xf - Xf_proj, proj


def _fit_theta_a_with_mode(
    *,
    Xf_all: np.ndarray,
    Xu_cur_all: np.ndarray,
    yp_all: np.ndarray,
    theta_g: np.ndarray,
    orth_mode: str,
    ridge_alpha_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    resid_after_g = yp_all - (Xu_cur_all @ np.asarray(theta_g, dtype=float))
    if orth_mode in {"plain", "weighted"}:
        Xf_fit, _proj = _orth_components(Xf_all, Xu_cur_all)
        theta_a = _ridge_solve(Xf_fit.T @ Xf_fit, Xf_fit.T @ resid_after_g, ridge_alpha=float(ridge_alpha_a))
        # Keep the runtime control head in the native coordinate system.
        # This preserves deployability while biasing the autonomous head away
        # from the control-reachable feature subspace.
        return np.asarray(theta_a, dtype=float), np.asarray(theta_g, dtype=float)
    theta_a = _ridge_solve(Xf_all.T @ Xf_all, Xf_all.T @ resid_after_g, ridge_alpha=float(ridge_alpha_a))
    return np.asarray(theta_a, dtype=float), np.asarray(theta_g, dtype=float)


def _svd_basis(kernels: Dict[str, np.ndarray], *, rank: int) -> np.ndarray:
    names = sorted(kernels)
    ref = np.asarray(kernels[names[0]], dtype=float)
    mats = []
    for name in names:
        h = np.asarray(kernels[name], dtype=float)
        if float(np.dot(h, ref)) < 0.0:
            h = -h
        mats.append(math.sqrt(float(TARGET_FIT_WEIGHTS.get(name, 1.0))) * h)
    K = np.vstack(mats)
    _, _, vt = np.linalg.svd(K, full_matrices=False)
    return np.asarray(vt[: int(rank), :], dtype=float)


def _project_kernel_to_basis(
    *,
    data: dict,
    basis: np.ndarray,
    ridge_alpha_g: float,
) -> tuple[np.ndarray, np.ndarray]:
    agg_XtX = float(data["fit_weight"]) * np.sum(np.stack(data["XtX_blocks"], axis=0), axis=0)
    agg_Xty = float(data["fit_weight"]) * np.sum(np.stack(data["Xty_blocks"], axis=0), axis=0)
    Bt = np.asarray(basis, dtype=float)
    A = Bt @ agg_XtX @ Bt.T + float(ridge_alpha_g) * np.eye(Bt.shape[0], dtype=float)
    b = Bt @ agg_Xty
    coeff = np.linalg.solve(A, b)
    h = Bt.T @ coeff
    return np.asarray(h, dtype=float), np.asarray(coeff, dtype=float)


def train_family(
    *,
    family: str,
    out_dir: Path,
    prbs_raw: Dict[tuple[int, int], base_train.RawRun],
    zero_raw: Dict[tuple[int, int], base_train.RawRun],
    winds: Sequence[int],
    seeds: Sequence[int],
    N: int,
    ts: float,
    t_start_s: float,
) -> None:
    cfg = family_cfg(family)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_cur, runtime_meta_by_ws = build_runs_by_wind(
        raw_lookup=prbs_raw,
        winds=winds,
        seeds=seeds,
        family=family,
        N=N,
        ts=ts,
        t_start_s=t_start_s,
    )
    runs_ref, _ = build_runs_by_wind(
        raw_lookup=zero_raw,
        winds=winds,
        seeds=seeds,
        family=family,
        N=N,
        ts=ts,
        t_start_s=t_start_s,
    )

    kernel_prior_penalty, kernel_prior_meta = _build_kernel_prior_penalty_matrix(
        N=int(N),
        u_mode=U_MODE,
        mode=str(cfg["gm_kernel_prior_mode"]),
        alpha=0.98,
        lambda_=float(cfg["gm_kernel_prior_lambda"]),
        eps=1e-9,
        normalize_mode="mean_diag",
    )

    manifest = []
    for ws in winds:
        runs_ws_by_target = {t: runs_cur.get(int(ws), {}).get(t, []) for t in TARGETS}
        if any(not v for v in runs_ws_by_target.values()):
            continue
        ref_lookup_by_target = {
            t: {(int(r.wind_mps), int(r.seed)): r for r in runs_ref.get(int(ws), {}).get(t, [])}
            for t in TARGETS
        }
        datasets = {
            t: _build_target_dataset(
                runs_ws=runs_ws_by_target[t],
                ref_lookup_ws=ref_lookup_by_target[t],
                target=t,
                N=int(N),
                ts=float(ts),
                t_start_s=float(t_start_s),
                horizon_weight_decay=float(cfg["horizon_weight_decay"]),
            )
            for t in TARGETS
        }
        indep_kernels = {
            t: _solve_indep_kernel(
                data=datasets[t],
                kernel_prior_penalty=kernel_prior_penalty,
                ridge_alpha_g=RIDGE_G,
            )
            for t in TARGETS
        }

        basis = None
        coeff_map: Dict[str, np.ndarray] = {}
        if str(cfg["kind"]) == "shared":
            basis = _svd_basis(indep_kernels, rank=int(cfg["shared_rank"]))

        for target in TARGETS:
            if basis is None:
                h_kernel = np.asarray(indep_kernels[target], dtype=float)
                coeff = np.asarray([], dtype=float)
            else:
                h_kernel, coeff = _project_kernel_to_basis(
                    data=datasets[target],
                    basis=basis,
                    ridge_alpha_g=RIDGE_G,
                )
            coeff_map[target] = coeff
            thetas_g_raw = _expand_toeplitz_kernel(h_kernel=h_kernel, N=int(N), u_mode=U_MODE)
            thetas_a: List[np.ndarray] = []
            thetas_g: List[np.ndarray] = []
            for Xf_all, Xu_cur_all, yp_all, _Xu_pair_all, _p in datasets[target]["current_designs"]:
                theta_idx = len(thetas_a)
                theta_a, theta_g = _fit_theta_a_with_mode(
                    Xf_all=Xf_all,
                    Xu_cur_all=Xu_cur_all,
                    yp_all=yp_all,
                    theta_g=np.asarray(thetas_g_raw[theta_idx], dtype=float),
                    orth_mode=str(cfg["orth_mode"]),
                    ridge_alpha_a=RIDGE_A,
                )
                thetas_a.append(theta_a)
                thetas_g.append(theta_g)
            thetas = [np.concatenate([th_a, th_g], axis=0) for th_a, th_g in zip(thetas_a, thetas_g)]
            model = {
                "target": str(target),
                "y_unit": "unit",
                "y_std": float(datasets[target]["y_std"]),
                "N": int(N),
                "ts": float(ts),
                "t_start_s": float(t_start_s),
                "ridge_alpha": float(RIDGE_A),
                "ridge_alpha_a": float(RIDGE_A),
                "ridge_alpha_g": float(RIDGE_G),
                "gm_lag_lambda": 0.0,
                "gm_kernel_prior_mode": str(cfg["gm_kernel_prior_mode"]),
                "gm_kernel_prior_lambda": float(cfg["gm_kernel_prior_lambda"]),
                "gm_kernel_prior_alpha": 0.98,
                "gm_kernel_prior_eps": 1e-9,
                "gm_kernel_prior_normalize": "mean_diag",
                "n_f": int(datasets[target]["n_f"]),
                "u_mode": U_MODE,
                "u_sched_ref": datasets[target]["u_sched_ref"],
                "fit_structure": "split_toeplitz",
                "gm_reference_mode": "paired_delta",
                "gm_u_source": GM_U_SOURCE,
                "gm_schedule_source": GM_SCHEDULE_SOURCE,
                "feature_mode": "algebraic_cleanup_proxy",
                "feature_runtime": runtime_meta_by_ws[int(ws)],
                "thetas": thetas,
                "thetas_a": thetas_a,
                "thetas_g": thetas_g,
                "theta_g_kernel": np.asarray(h_kernel, dtype=float),
                "g_kernel_u_blocks": int(u_blocks_for_mode(U_MODE)),
                "g_kernel_N": int(N),
                "phase31_family": str(family),
                "phase31_kind": str(cfg["kind"]),
                "phase31_orth_mode": str(cfg["orth_mode"]),
                "phase31_horizon_weight_decay": float(cfg["horizon_weight_decay"]),
                "phase31_shared_rank": int(cfg["shared_rank"]),
                "phase31_shared_targets": list(TARGETS),
                "phase31_shared_basis": [] if basis is None else np.asarray(basis, dtype=float).tolist(),
                "phase31_shared_coeff": np.asarray(coeff_map[target], dtype=float).tolist(),
                "phase31_target_fit_weight": float(TARGET_FIT_WEIGHTS[target]),
                "gm_kernel_prior": dict(kernel_prior_meta) if kernel_prior_meta is not None else None,
            }
            base_train._save_model(
                model,
                out_path=out_dir / f"direct_ws{int(ws):02d}_{target}.npz",
                feature_names=list(feature_names_for_family_target(family, target)),
            )
            print(
                f"[done] family={family} ws={ws} target={target} kind={cfg['kind']} orth={cfg['orth_mode']} rank={cfg['shared_rank']}",
                flush=True,
            )
        manifest.append(
            {
                "family": str(family),
                "wind_mps": int(ws),
                "kind": str(cfg["kind"]),
                "orth_mode": str(cfg["orth_mode"]),
                "shared_rank": int(cfg["shared_rank"]),
                "horizon_weight_decay": float(cfg["horizon_weight_decay"]),
            }
        )
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
