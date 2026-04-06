# Omitted Diff Excerpts

This file records only the core raw code excerpts for files omitted from this compressed archive.

Retained full references in this folder are:

- `baseline__old_unified__direct_mstep_features.py`
- `baseline__proxy21d__algebraic_cleanup_proxy.py`
- `step1__phase3_delta_u_proxy.py`
- `step2__phase20_delta_u_proxy.py`
- `step2__phase20_zmq_mpc_server_phase20.py`
- `step3__phase31_proxy.py`
- `step3__phase31_sharedgeometry_train_utils.py`

All later server files share the same main `phase20` controller backbone. The excerpts below only preserve the delta blocks that motivated each omitted file.

## Step 1 Omitted Files

### `phase2_control_leakage_2026_03_27/control_leakage_proxy.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase2_control_leakage_2026_03_27/control_leakage_proxy.py`

```python
ROUTED_FEATURES = dict(_base.ROUTED_FEATURES)
ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"] = _drop(
    ROUTED_FEATURES["routed_fa_prunedwork"],
    CMD_MEMORY_FEATURES,
)
ROUTED_FEATURES["routed_twr_prunedwork_nocmdmem"] = _drop(
    ROUTED_FEATURES["routed_twr_prunedwork"],
    CMD_MEMORY_FEATURES,
)
ROUTED_FEATURES["routed_fa_prunedwork_noactmem"] = _drop(
    ROUTED_FEATURES["routed_fa_prunedwork"],
    ACTUATOR_MEMORY_FEATURES,
)
ROUTED_FEATURES["routed_twr_prunedwork_noactmem"] = _drop(
    ROUTED_FEATURES["routed_twr_prunedwork"],
    ACTUATOR_MEMORY_FEATURES,
)
```

### `phase3_delta_u_consistency_2026_03_27/zmq_mpc_server_delta.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase3_delta_u_consistency_2026_03_27/zmq_mpc_server_delta.py`

```python
if decision_coords == "delta_u":
    P_du_chain = np.zeros((n_u, n_u), dtype=float)
    if n_u > 1:
        P_du_chain[1:, 1:] = 2.0 * float(w_du_chain) * np.eye(n_u - 1, dtype=float)
    P_du_anchor = 2.0 * float(w_du_anchor) * np.outer(anchor_vec, anchor_vec)
else:
    P_du_chain = 2.0 * float(w_du_chain) * (D_chain.T @ D_chain) if D_chain.size else np.zeros((n_u, n_u), dtype=float)
    P_du_anchor = 2.0 * float(w_du_anchor) * np.outer(anchor_vec, anchor_vec)

P_abs_u = P_u + P_ufreq
if decision_coords == "delta_u":
    P_u_geom = C_u.T @ P_abs_u @ C_u
else:
    P_u_geom = P_abs_u
```

### `phase4_control_orthogonalization_2026_03_27/orth_train_utils.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase4_control_orthogonalization_2026_03_27/orth_train_utils.py`

```python
if str(orth_mode) == "xgram":
    XtX_a = Xf_all.T @ Xf_all
    Xty_a = Xf_all.T @ yp_all
    cross = (Xu_cur_all.T @ Xf_all) / max(1.0, float(Xf_all.shape[0]))
    penalty = float(xgram_lambda) * (cross.T @ cross)
    theta_a = _ridge_solve_augmented(
        XtX_a,
        Xty_a,
        ridge_alpha=float(ridge_alpha_a),
        extra_penalty=penalty,
    )
    thetas_a.append(np.asarray(theta_a, dtype=float))
...
if str(orth_mode) == "gfirst":
    thetas_a = []
    for (Xf_all, Xu_cur_all, yp_all), theta_g in zip(current_designs, thetas_g):
        resid_a = yp_all - (Xu_cur_all @ np.asarray(theta_g, dtype=float))
        theta_a = _ridge_solve(
            Xf_all.T @ Xf_all,
            Xf_all.T @ resid_a,
            ridge_alpha=float(ridge_alpha_a),
```

### `phase5_actuator_state_2026_03_27/actstate_train_utils.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase5_actuator_state_2026_03_27/actstate_train_utils.py`

```python
if str(gm_u_source) == "pit_offset":
    u_series = r.u
elif str(gm_u_source) == "actuator_delta":
    n_pair = min(len(r.beta0), len(r_ref.beta0))
    u_series = r.beta0[:n_pair] - r_ref.beta0[:n_pair]
else:
    raise ValueError(f"unknown gm_u_source={gm_u_source!r}")
uk_pair = np.column_stack([u_series[i0 + j : k_max + j] for j in range(int(p))])
if str(gm_schedule_source) == "reference":
    omega_src = r_ref.omega0[i0:k_max]
    beta_src = r_ref.beta0[i0:k_max]
    wind_src = r_ref.wind0[i0:k_max]
```

### `phase6_gm_prior_2026_03_27/delta_u_proxy.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase6_gm_prior_2026_03_27/delta_u_proxy.py`

```python
ROUTED_FEATURES = dict(_base.ROUTED_FEATURES)
ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"] = _drop(
    ROUTED_FEATURES["routed_fa_prunedwork"],
    CMD_MEMORY_FEATURES,
)

def feature_names_for_family_target(family: str, target: str) -> tuple[str, ...]:
    if family in {
        "deploy_acc_nocmd_pitoffset_stabletc1",
        "deploy_acc_nocmd_actdelta_stabletc03",
        "deploy_acc_nocmd_actdelta_stabletc1",
        "deploy_acc_nocmd_actdelta_stabletc3",
    }:
        return ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"]
```

### `phase7_shared_kernel_2026_03_27/sharedkernel_train_utils.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase7_shared_kernel_2026_03_27/sharedkernel_train_utils.py`

```python
def _fit_shared_kernel_family(
    *,
    family: str,
    runs_ws_by_target: Dict[str, List[RunData]],
    ref_lookup_ws_by_target: Dict[str, Dict[tuple[int, int], RunData]],
    runs_by_wind_by_target: Dict[str, Dict[int, List[RunData]]],
    current_ws: int,
    N: int,
    ts: float,
    t_start_s: float,
    ridge_alpha_a: float,
    ridge_alpha_g: float,
    u_mode: str,
    gm_schedule_source: str,
) -> dict[str, dict]:
    shared_targets = [t for t in SHARED_TARGETS if runs_ws_by_target.get(t)]
    if len(shared_targets) < 2:
        raise RuntimeError("shared-kernel training requires at least two targets")
```

### `phase8_bootstrap_uncertainty_2026_03_27/zmq_mpc_server_phase8.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase8_bootstrap_uncertainty_2026_03_27/zmq_mpc_server_phase8.py`

```python
def _uncertainty_shaped_B(
    *,
    kernel: np.ndarray,
    rel_std: np.ndarray,
    gamma: float,
    n_f: int,
    N: int,
    u_mode: str,
    y_std: float,
    floor: float = 0.2,
) -> tuple[list[np.ndarray], np.ndarray]:
    ...
    conf = 1.0 / (1.0 + gamma * np.square(np.clip(rel_std, 0.0, 50.0)))
    conf = np.clip(conf, floor, 1.0)
    kernel_unc = kernel * conf
    thetas_g_unc = _expand_toeplitz_kernel(
        h_kernel=np.asarray(kernel_unc, dtype=float),
        N=int(N),
        u_mode=str(u_mode),
    )
```

### `phase10_power_state_ablation_2026_03_28/delta_u_proxy.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase10_power_state_ablation_2026_03_28/delta_u_proxy.py`

```python
POWER_STATE_FEATURES = (
    "GenSpeed",
    "GenTqMeas",
    "VS_GenPwr_W",
)

ROUTED_FEATURES["power_prunedwork_all3"] = _extend_unique(
    ROUTED_FEATURES["power_prunedwork_base"],
    POWER_STATE_FEATURES,
)

def feature_names_for_family_target(family: str, target: str) -> tuple[str, ...]:
    power_key = FAMILY_POWER_FEATURE_KEY.get(family)
    if power_key is not None:
        if target == "GenPwr":
            return ROUTED_FEATURES[power_key]
        return ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"]
```

### `phase10_power_state_ablation_2026_03_28/zmq_mpc_server_phase10.py`
Source:
`Exp/control_geometry_consistency_program_2026-03-27/code/phase10_power_state_ablation_2026_03_28/zmq_mpc_server_phase10.py`

```python
if secondary_target is not None:
    secondary_meta, secondary_thetas = _load_model(args.models_dir, wind_mps=wind_mps, target=str(secondary_target))
    if list(secondary_meta["feature_names"]) != list(feature_names):
        raise SystemExit("secondary model feature_names mismatch")
    if int(secondary_meta["meta"]["n_f"]) != int(n_f):
        raise SystemExit("secondary model n_f mismatch")
    if str(secondary_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
        raise SystemExit("secondary model feature_mode mismatch")
    secondary_u_mode = str(secondary_meta["meta"].get("u_mode", "fixed"))
```

## Step 2 Omitted Files

### `phase13_peak_sensitive_poweraware_2026_03_28/run_phase13_peak_sensitive_poweraware.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase13_peak_sensitive_poweraware_2026_03_28/run_phase13_peak_sensitive_poweraware.py`

```python
def _shape_args() -> list[str]:
    return [
        "--mpc-tower-cone-alpha-band",
        "1.0",
        "--mpc-tower-cone-alpha-envlp95",
        "0.30",
        "--mpc-tower-cone-alpha-tp90",
        "0.15",
        "--mpc-tower-cone-alpha-sb3p-env95",
        "0.05",
    ]
...
{
    "label": "phase13_cap120_f10_delta_old",
    "models_dir": PHASE10_MODELS,
    "args": [
```

### `phase14_output_constrained_collective_2026_03_28/zmq_mpc_server_phase14.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase14_output_constrained_collective_2026_03_28/zmq_mpc_server_phase14.py`

```python
if (
    qp.idx_secondary_cap is not None
    and qp.n_secondary_cap > 0
    and float(secondary_cap_rel) > 0.0
    and a_load_secondary is not None
    and B_qp_secondary is not None
):
    if str(args.secondary_cap_mode) == "identity":
        H_sec = np.eye(int(qp.n_u), dtype=float)
        A_cap = B_qp_secondary
        z_free = a_load_secondary
        free_rms = float(np.sqrt(np.mean(np.square(z_free)))) if z_free.size else 0.0
        cap_floor = max(float(args.secondary_cap_floor), float(args.secondary_cap_floor_rel) * max(free_rms, 1e-9), 1e-6)
```

### `phase15_schedule_cleanup_2026_03_28/run_phase15_schedule_cleanup.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase15_schedule_cleanup_2026_03_28/run_phase15_schedule_cleanup.py`

```python
def _variant_specs() -> list[dict[str, object]]:
    return [
        {
            "label": "phase15_budget_auto1p3p",
            "args": [],
        },
        {
            "label": "phase15_budget_auto1p3p_half015",
            "args": [
                "--mpc-u-freq-halfwidth-hz",
                "0.015",
```

### `phase17_applied_channel_id_2026_03_28/algebraic_cleanup_proxy.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase17_applied_channel_id_2026_03_28/algebraic_cleanup_proxy.py`

```python
def make_algebraic_cleanup_params(*, applied_state: Mapping[str, float] | None = None) -> dict:
    params = dict(_alg_base.make_algebraic_cleanup_params())
    if applied_state is not None:
        for key, val in dict(applied_state).items():
            params[str(key)] = float(val)
    return params

def _update_state(x_prev: float, u_prev: float, *, mode: str, params: Mapping[str, object]) -> float:
    if str(mode) == "idstate1":
        a = float(params.get("applied_id1_a", 0.0))
        b = float(params.get("applied_id1_b", 1.0))
        return float(a * x_prev + b * u_prev)
```

### `phase17_applied_channel_id_2026_03_28/zmq_mpc_server_phase17.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase17_applied_channel_id_2026_03_28/zmq_mpc_server_phase17.py`

```python
if secondary_target is not None:
    secondary_meta, secondary_thetas = _load_model(args.models_dir, wind_mps=wind_mps, target=str(secondary_target))
    if list(secondary_meta["feature_names"]) != list(feature_names):
        raise SystemExit("secondary model feature_names mismatch")
    if int(secondary_meta["meta"]["n_f"]) != int(n_f):
        raise SystemExit("secondary model n_f mismatch")
    if str(secondary_meta["meta"].get("feature_mode", feature_mode)) != feature_mode:
        raise SystemExit("secondary model feature_mode mismatch")
```

### `phase18_shared_kernel_blocksep_2026_03_28/sharedkernel_train_utils.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase18_shared_kernel_blocksep_2026_03_28/sharedkernel_train_utils.py`

```python
def _fit_shared_kernel_family(
    *,
    family: str,
    runs_ws_by_target: Dict[str, List[RunData]],
    ref_lookup_ws_by_target: Dict[str, Dict[tuple[int, int], RunData]],
    runs_by_wind_by_target: Dict[str, Dict[int, List[RunData]]],
    current_ws: int,
    N: int,
    ts: float,
    t_start_s: float,
    ridge_alpha_a: float,
    ridge_alpha_g: float,
    u_mode: str,
    gm_schedule_source: str,
) -> dict[str, dict]:
    shared_targets = [t for t in SHARED_TARGETS if runs_ws_by_target.get(t)]
```

### `phase19_controllability_basis_2026_03_29/zmq_mpc_server_phase19.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase19_controllability_basis_2026_03_29/zmq_mpc_server_phase19.py`

```python
def _build_controllability_basis(
    *,
    K_useful: np.ndarray,
    K_side: np.ndarray,
    rank: int,
    side_eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ...
    G_eff = side_inv_sqrt @ K_useful @ side_inv_sqrt
    G_eff = 0.5 * (G_eff + G_eff.T)
    eval_eff, U_eff = np.linalg.eigh(G_eff)
...
V_basis, control_basis_stats = _build_controllability_basis(
    K_useful=K_useful,
    K_side=K_side,
    rank=int(args.control_basis_rank),
    side_eps=float(args.control_basis_side_eps),
)
```

### `phase21_mbc_pilot_2026_03_29/zmq_mpc_server_phase21.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase21_mbc_pilot_2026_03_29/zmq_mpc_server_phase21.py`

```python
blade_phases = _mbc_blade_phases_rad(float(meas.get("Azimuth", 0.0)))
cyc_raw = _mbc_cyclic_offsets(float(u1c_cmd), float(u1s_cmd), blade_phases)
mbc_alpha = _max_feasible_cyclic_scale(
    collective_cmd=float(u_cmd),
    cyclic_offsets=cyc_raw,
    blade_prev=blade_prev,
    u_abs_max=float(u_max_runtime),
    du_abs_max=float(du_max_runtime),
)
u1c_cmd *= float(mbc_alpha)
u1s_cmd *= float(mbc_alpha)
blade_cmds = float(u_cmd) + _mbc_cyclic_offsets(float(u1c_cmd), float(u1s_cmd), blade_phases)
```

### `phase22_collective_cyclic_lexi_2026_03_29/zmq_mpc_server_phase22.py`
Source:
`Exp/control_geometry_step2_program_2026-03-28/code/phase22_collective_cyclic_lexi_2026_03_29/zmq_mpc_server_phase22.py`

```python
a_root1c = qp_root1c.Af_load @ f_k_root1c
B_root1c = _effective_B(
    blocks=qp_root1c.B_blocks_load,
    u_mode=qp_root1c.u_mode_load,
    sched_ref=qp_root1c.u_sched_ref_load,
    meas=meas,
    wind_mps=wind_mps,
)
...
res1s = _solve_qp_bundle(qp=qp_root1s, P_mat=P_root1s, q_vec=q_root1s, l=l1s, u=u1s_bounds, A_rebuild_dense=A1s_rebuild)
if res1s.x is not None and "solved" in status1s.lower():
    if str(qp_root1s.decision_coords) == "delta_u":
        v1s = np.asarray(res1s.x[: qp_root1s.n_u], dtype=float)
        u1s_abs = float(u1s_prev) * qp_root1s.ones_u + qp_root1s.C_u @ v1s
        u1s_cmd = float(u1s_abs[0])
```

## Step 3 Omitted Files

### `phase32_mode_tagged_priors_2026_03_30/phase32_proxy.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase32_mode_tagged_priors_2026_03_30/phase32_proxy.py`

```python
"phase32_indep_towertag": {
    "kind": "indep",
    "orth_mode": "none",
    "horizon_weight_decay": 0.0,
    "prior_scheme": "tower_tagged",
},
"phase32_indep_roottag": {
    "kind": "indep",
    "orth_mode": "none",
    "horizon_weight_decay": 0.0,
    "prior_scheme": "root_tagged",
},
```

### `phase32_mode_tagged_priors_2026_03_30/modetagged_train_utils.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase32_mode_tagged_priors_2026_03_30/modetagged_train_utils.py`

```python
for comp in comps:
    kind = str(comp["kind"])
    if kind == "stable_tc":
        block, block_meta = _build_kernel_prior_block(
            N=int(N),
            mode="stable_tc",
            alpha=float(comp.get("alpha", 0.98)),
            eps=1e-9,
            normalize_mode="mean_diag",
        )
        Q = _kron_eye(block, u_mode=U_MODE)
        meta_row = {"kind": kind, **block_meta}
```

### `phase33_reachability_split_2026_03_30/zmq_mpc_server_phase33.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase33_reachability_split_2026_03_30/zmq_mpc_server_phase33.py`

```python
blade_phases = _mbc_blade_phases_rad(float(meas.get("Azimuth", 0.0)))
cyc_raw = _mbc_cyclic_offsets(float(u1c_cmd), float(u1s_cmd), blade_phases)
mbc_alpha = _max_feasible_cyclic_scale(
    collective_cmd=float(u_cmd),
    cyclic_offsets=cyc_raw,
    blade_prev=blade_prev,
    u_abs_max=float(u_max_runtime),
    du_abs_max=float(du_max_runtime),
)
u1c_cmd *= float(mbc_alpha)
u1s_cmd *= float(mbc_alpha)
blade_cmds = float(u_cmd) + _mbc_cyclic_offsets(float(u1c_cmd), float(u1s_cmd), blade_phases)
...
out[2] = float(blade_cmds[0])
out[3] = float(blade_cmds[1])
out[4] = float(blade_cmds[2])
```

### `phase34_conflict_aware_basis_2026_03_30/zmq_mpc_server_phase34.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase34_conflict_aware_basis_2026_03_30/zmq_mpc_server_phase34.py`

```python
def _build_conflict_aware_basis(
    *,
    K_useful: np.ndarray,
    K_health: Optional[np.ndarray],
    K_speed: Optional[np.ndarray],
    K_power: Optional[np.ndarray],
    K_root: Optional[np.ndarray],
    rank: int,
    side_eps: float,
    health_weight: float,
    speed_weight: float,
    power_weight: float,
    root_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ...
    K_side = (
        float(health_weight) * _normalize_psd_component(K_ref, K_health_raw)
        + float(speed_weight) * _normalize_psd_component(K_ref, K_speed_raw)
        + float(power_weight) * _normalize_psd_component(K_ref, K_power_raw)
        + float(root_weight) * _normalize_psd_component(K_ref, K_root_raw)
    )
```

### `phase35_collective_torque_hierarchy_2026_03_30/zmq_mpc_server_phase35.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase35_collective_torque_hierarchy_2026_03_30/zmq_mpc_server_phase35.py`

```python
def _augment_phase35_torque_meas(
    meas: Dict[str, float],
    *,
    tq_hist: deque[float],
    pwr_hist: deque[float],
    rot_hist: deque[float],
) -> Dict[str, float]:
    ...
    out["RotEnergy"] = float(rot * rot)
    out["HorWindV2"] = float(wind * wind)
    out["HorWindV_RotSpeed"] = float(wind * rot)
    out["RotSpeed_GenTqMeas"] = float(rot * tq)
...
def _solve_scalar_torque_channel(
    *,
    a_speed: Optional[np.ndarray],
    b_speed: Optional[np.ndarray],
    a_pwr: Optional[np.ndarray],
    b_pwr: Optional[np.ndarray],
    prev_nm: float,
```

### `phase36_harmonic_cyclic_2026_03_31/zmq_mpc_server_phase36.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase36_harmonic_cyclic_2026_03_31/zmq_mpc_server_phase36.py`

```python
ap.add_argument(
    "--mbc-harmonic-mode",
    choices=["free_sequence", "harmonic_1p_const"],
    default="free_sequence",
    help="How cyclic MBC commands are parameterized: free_sequence matches the old pilot; harmonic_1p_const constrains each fixed-frame channel to a single constant 1P amplitude over the horizon.",
)
ap.add_argument("--torque-enable", action="store_true")
ap.add_argument("--torque-models-dir", type=Path, default=None)
...
tq_stats = _solve_scalar_torque_channel(
    a_speed=a_torque_speed,
    b_speed=b_torque_speed,
    a_pwr=a_torque_pwr,
    b_pwr=b_torque_pwr,
```

### `phase37_actuator_health_2026_03_31/zmq_mpc_server_phase37.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase37_actuator_health_2026_03_31/zmq_mpc_server_phase37.py`

```python
health_penalty_scale = 1.0
health_limit_scale = 1.0
health_wload_scale = 1.0
if bool(args.health_enable):
    health_penalty_scale = 1.0 + float(args.health_stage_gain) * float(health_total) + float(args.health_reversal_gain) * float(health_reversal_state)
    if str(args.health_mode) in ("budget", "lexi"):
        health_limit_scale = 1.0 / (1.0 + float(args.health_budget_gain) * float(health_budget))
        health_limit_scale = float(np.clip(health_limit_scale, float(args.health_budget_floor), 1.0))
        u_max_runtime *= float(health_limit_scale)
        du_max_runtime *= float(health_limit_scale)
```

### `phase38_disturbance_preview_2026_03_31/delta_u_proxy.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase38_disturbance_preview_2026_03_31/delta_u_proxy.py`

```python
if family in {"deploy_disturb_collective_stabletc3_all3", "deploy_disturb_combined_stabletc3_all3", "deploy_disturb_preview_stabletc3_all3"}:
    if target in {"FA_Acc_TT", "GenPwr", "RotSpeed", "RootM0"}:
        feats = _extend_unique(feats, COLLECTIVE_DIST_FEATURES)
if family in {"deploy_disturb_cyclic_stabletc3_all3", "deploy_disturb_combined_stabletc3_all3", "deploy_disturb_preview_stabletc3_all3"}:
    if target in {"RootM1c", "RootM1s"}:
        feats = _extend_unique(feats, CYCLIC_DIST_FEATURES)
if family == "deploy_disturb_preview_stabletc3_all3":
    if target in {"FA_Acc_TT", "GenPwr", "RotSpeed", "RootM0"}:
        feats = _extend_unique(feats, PREVIEW_DIST_FEATURES)
```

### `phase38_disturbance_preview_2026_03_31/zmq_mpc_server_phase38.py`
Source:
`Exp/control_geometry_step3_program_2026-03-30/code/phase38_disturbance_preview_2026_03_31/zmq_mpc_server_phase38.py`

```python
def _augment_phase38_disturbance_meas(
    meas: Dict[str, float],
    *,
    wind_hist16: deque[float],
    wind_hist32: deque[float],
    dv1s_hist16: deque[float],
    dv1c_hist16: deque[float],
) -> Dict[str, float]:
    ...
    out["TSRProxy"] = float(rot / max(abs(wind), 1e-3))
    out["WindMean16"] = _hist_mean(wind_hist16)
    out["WindMean32"] = _hist_mean(wind_hist32)
    out["WindTrend32"] = float(wind_trend32)
...
if feature_mode == "phase38_disturbance_proxy":
    meas_feature = _augment_phase38_disturbance_meas(
        meas_feature,
        wind_hist16=wind_hist16,
        wind_hist32=wind_hist32,
        dv1s_hist16=dv1s_hist16,
        dv1c_hist16=dv1c_hist16,
    )
```
