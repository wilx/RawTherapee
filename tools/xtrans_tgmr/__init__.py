"""Phase-conditioned Student-t mixture-regression research tools."""

from .model import (
    StudentTBatch,
    StudentTTrainingResult,
    conditional_student_t_predict,
    estimate_shared_degrees_of_freedom,
    refine_fixed_student_t_mixture,
    refine_learned_student_t_mixture,
    student_t_logpdf,
)

__all__ = (
    "StudentTBatch",
    "StudentTTrainingResult",
    "conditional_student_t_predict",
    "estimate_shared_degrees_of_freedom",
    "refine_fixed_student_t_mixture",
    "refine_learned_student_t_mixture",
    "student_t_logpdf",
)
