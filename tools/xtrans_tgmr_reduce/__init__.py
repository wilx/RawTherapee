"""Model-reduction tools for the phase-conditioned Student-t GMR."""

from .model import (
    FactorCache,
    FactorModel,
    ShortlistBatch,
    approximate_factor_model,
    coarse_component_order,
    factor_student_t_predict,
    prepare_factor_cache,
    prune_components,
    shortlist_student_t_predict,
    truncate_exact_posterior,
)

__all__ = (
    "FactorCache",
    "FactorModel",
    "ShortlistBatch",
    "approximate_factor_model",
    "coarse_component_order",
    "factor_student_t_predict",
    "prepare_factor_cache",
    "prune_components",
    "shortlist_student_t_predict",
    "truncate_exact_posterior",
)
