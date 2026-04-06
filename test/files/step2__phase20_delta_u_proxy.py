from __future__ import annotations

import sys
from pathlib import Path


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
REPO = PROGRAM_ROOT.parents[1]
BASE_ROOT = REPO / "Exp" / "algebraic_cleanup_probe_2026-03-25"
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

import algebraic_cleanup_proxy as _base  # noqa: E402


CMD_MEMORY_FEATURES = (
    "PitOffsetApplied_c_lag01",
    "PitOffsetApplied_du_lag01",
)

POWER_STATE_FEATURES = (
    "GenSpeed",
    "GenTqMeas",
    "VS_GenPwr_W",
)


def _drop(features: tuple[str, ...], blocked: tuple[str, ...]) -> tuple[str, ...]:
    blocked_set = set(str(x) for x in blocked)
    return tuple(name for name in features if str(name) not in blocked_set)


def _extend_unique(features: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    out = list(features)
    seen = set(out)
    for name in extra:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return tuple(out)


ROUTED_FEATURES = dict(_base.ROUTED_FEATURES)
ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"] = _drop(
    ROUTED_FEATURES["routed_fa_prunedwork"],
    CMD_MEMORY_FEATURES,
)
ROUTED_FEATURES["power_prunedwork_base"] = ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"]
ROUTED_FEATURES["power_prunedwork_genspeed"] = _extend_unique(
    ROUTED_FEATURES["power_prunedwork_base"],
    ("GenSpeed",),
)
ROUTED_FEATURES["power_prunedwork_gentq"] = _extend_unique(
    ROUTED_FEATURES["power_prunedwork_base"],
    ("GenTqMeas",),
)
ROUTED_FEATURES["power_prunedwork_vspwr"] = _extend_unique(
    ROUTED_FEATURES["power_prunedwork_base"],
    ("VS_GenPwr_W",),
)
ROUTED_FEATURES["power_prunedwork_all3"] = _extend_unique(
    ROUTED_FEATURES["power_prunedwork_base"],
    POWER_STATE_FEATURES,
)
ROUTED_FEATURES["power_oldphys16"] = (
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
)


FAMILY_POWER_FEATURE_KEY = {
    "deploy_acc_nocmd_actdelta_stabletc1_pwr_all3": "power_prunedwork_all3",
    "deploy_acc_nocmd_actdelta_stabletc3_pwr_all3": "power_prunedwork_all3",
    "deploy_acc_nocmd_actdelta_stabletc6_pwr_all3": "power_prunedwork_all3",
}


def feature_names_for_family_target(family: str, target: str) -> tuple[str, ...]:
    power_key = FAMILY_POWER_FEATURE_KEY.get(family)
    if power_key is not None:
        if target == "GenPwr":
            return ROUTED_FEATURES[power_key]
        return ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"]
    return _base.feature_names_for_family_target(family, target)


def family_feature_map() -> dict[str, tuple[str, ...]]:
    feats = ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"]
    return {
        "deploy_acc_nocmd_actdelta_stabletc1_pwr_all3": feats,
        "deploy_acc_nocmd_actdelta_stabletc3_pwr_all3": feats,
        "deploy_acc_nocmd_actdelta_stabletc6_pwr_all3": feats,
    }


build_load_projection = _base.build_load_projection
compute_algebraic_cleanup_refs = _base.compute_algebraic_cleanup_refs
is_algebraic_cleanup_proxy_mode = _base.is_algebraic_cleanup_proxy_mode
make_algebraic_cleanup_params = _base.make_algebraic_cleanup_params
simulate_algebraic_cleanup_df = _base.simulate_algebraic_cleanup_df
