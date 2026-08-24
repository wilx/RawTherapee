"""Phase-conditioned Gaussian-mixture regression for X-Trans research."""

from .model import (
    PhaseConditionedGMR,
    conditional_predict,
    fit_phase_conditioned_gmr,
    prepare_gmr_cache,
)

__all__ = (
    "PhaseConditionedGMR",
    "conditional_predict",
    "fit_phase_conditioned_gmr",
    "prepare_gmr_cache",
)
