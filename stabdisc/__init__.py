"""Reproducible success probabilities for adaptive stabilizer discrimination."""

from .algebra import Qsqrt2
from .ensemble import Ensemble
from .numeric import density_matrices_to_ensemble, ensemble_to_density_matrices
from .solver import SolveResult, SolverLimitError, StabilizerDiscriminationSolver, success_probability
from .states import (append_t_states, e6_full, e6_sign, e8_full, e8_sign,
                     four_state, tensor_ensembles, twoqubit_mixture, xor_repetition)

__version__ = "0.4.0"
__all__ = [
    "Qsqrt2", "Ensemble", "SolveResult", "SolverLimitError",
    "StabilizerDiscriminationSolver", "success_probability",
    "density_matrices_to_ensemble", "ensemble_to_density_matrices",
    "append_t_states", "tensor_ensembles", "xor_repetition",
    "e8_full", "e8_sign", "e6_full", "e6_sign", "twoqubit_mixture", "four_state",
]
