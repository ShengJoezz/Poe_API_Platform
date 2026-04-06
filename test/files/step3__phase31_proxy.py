from __future__ import annotations

import math
import sys
from pathlib import Path


PROGRAM_ROOT = Path(__file__).resolve().parents[2]
REPO = PROGRAM_ROOT.parents[1]
STEP2_ROOT = REPO / "Exp" / "control_geometry_step2_program_2026-03-28"
PHASE20_CODE = STEP2_ROOT / "code" / "phase20_modelset_robust_2026_03_29"
PHASE12_CODE = STEP2_ROOT / "code" / "phase12_aux_visibility_poweraware_2026_03_28"
for p in (PHASE20_CODE, PHASE12_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from delta_u_proxy import (  # type: ignore  # noqa: E402
    ROUTED_FEATURES as PHASE20_ROUTED_FEATURES,
    make_algebraic_cleanup_params,
)


TARGETS = (
    "FA_Acc_TT",
    "TwrBsMyt",
    "RotSpeed",
    "GenPwr",
    "RootM0",
    "RootM1c",
    "RootM1s",
)

BASE_STRUCTURAL = tuple(PHASE20_ROUTED_FEATURES["routed_fa_prunedwork_nocmdmem"])
POWER_ALL3 = tuple(PHASE20_ROUTED_FEATURES["power_prunedwork_all3"])

TARGET_FEATURES = {
    "FA_Acc_TT": BASE_STRUCTURAL,
    "TwrBsMyt": BASE_STRUCTURAL,
    "RotSpeed": BASE_STRUCTURAL + ("GenSpeed", "GenTqMeas", "VS_GenPwr", "RotSpeed_ma8", "RotSpeed_trend16"),
    "GenPwr": POWER_ALL3,
    "RootM0": BASE_STRUCTURAL + ("RootM0", "RootM0_ma8", "RootM0_trend16"),
    "RootM1c": BASE_STRUCTURAL + ("RootM1c", "RootM1s", "RootM1_amp"),
    "RootM1s": BASE_STRUCTURAL + ("RootM1c", "RootM1s", "RootM1_amp"),
}

FAMILIES = {
    "phase31_indep_phase20_all3": {
        "kind": "indep",
        "orth_mode": "none",
        "shared_rank": 0,
        "horizon_weight_decay": 0.0,
        "gm_kernel_prior_mode": "stable_tc",
        "gm_kernel_prior_lambda": 1.0,
    },
    "phase31_shared_plain_rank1": {
        "kind": "shared",
        "orth_mode": "none",
        "shared_rank": 1,
        "horizon_weight_decay": 0.0,
        "gm_kernel_prior_mode": "stable_tc",
        "gm_kernel_prior_lambda": 1.0,
    },
    "phase31_shared_orth_rank1": {
        "kind": "shared",
        "orth_mode": "plain",
        "shared_rank": 1,
        "horizon_weight_decay": 0.0,
        "gm_kernel_prior_mode": "stable_tc",
        "gm_kernel_prior_lambda": 1.0,
    },
    "phase31_shared_orthw_rank1": {
        "kind": "shared",
        "orth_mode": "weighted",
        "shared_rank": 1,
        "horizon_weight_decay": 0.18,
        "gm_kernel_prior_mode": "stable_tc",
        "gm_kernel_prior_lambda": 1.0,
    },
    "phase31_shared_orth_rank2": {
        "kind": "shared",
        "orth_mode": "plain",
        "shared_rank": 2,
        "horizon_weight_decay": 0.0,
        "gm_kernel_prior_mode": "stable_tc",
        "gm_kernel_prior_lambda": 1.0,
    },
}

DEFAULT_FAMILIES = tuple(FAMILIES.keys())

TARGET_FIT_WEIGHTS = {
    "FA_Acc_TT": 1.0,
    "TwrBsMyt": 1.0,
    "RotSpeed": 1.0,
    "GenPwr": 0.9,
    "RootM0": 0.9,
    "RootM1c": 0.8,
    "RootM1s": 0.8,
}


def family_feature_map() -> dict[str, tuple[str, ...]]:
    return {family: BASE_STRUCTURAL for family in DEFAULT_FAMILIES}


def family_cfg(family: str) -> dict:
    if family not in FAMILIES:
        raise KeyError(family)
    return dict(FAMILIES[family])


def feature_names_for_family_target(family: str, target: str) -> tuple[str, ...]:
    if family not in FAMILIES:
        raise KeyError(family)
    if target not in TARGET_FEATURES:
        raise KeyError(target)
    return tuple(TARGET_FEATURES[target])


def horizon_weight(p: int, *, decay: float) -> float:
    return float(math.exp(-float(decay) * max(int(p) - 1, 0)))
