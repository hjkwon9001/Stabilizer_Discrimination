"""Validated dense-matrix conversions (floating-point, not exact certificates)."""

from __future__ import annotations

import numpy as np

from .ensemble import Ensemble
from .pauli import unsigned_to_pauli_string

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def pauli_matrix(label):
    """Dense Pauli, with the first label character on the left tensor factor."""
    matrix = np.ones((1, 1), dtype=complex)
    for character in label:
        matrix = np.kron(matrix, PAULI[character.upper()])
    return matrix


def validate_density_matrix(rho, *, tol=1e-10):
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not rho.shape[0]:
        raise ValueError("density matrix must be nonempty and square")
    d = rho.shape[0]
    if d & (d - 1):
        raise ValueError("dimension must be a power of two")
    if not np.isfinite(rho).all():
        raise ValueError("density matrix must contain finite entries")
    if not np.allclose(rho, rho.conj().T, rtol=0, atol=tol):
        raise ValueError("density matrix must be Hermitian")
    if abs(np.trace(rho) - 1) > tol:
        raise ValueError("density matrix must have trace one")
    if np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() < -tol:
        raise ValueError("density matrix must be positive semidefinite")
    return rho


def density_matrices_to_ensemble(rhos, priors=None, labels=None, tol=1e-10):
    """Validate states and convert to floating Pauli expectations.

    No small coefficients are discarded. The tolerance is used only for
    validation, not as an optimization pruning rule or error certificate.
    """
    rhos = tuple(validate_density_matrix(rho, tol=tol) for rho in rhos)
    if not rhos:
        raise ValueError("rhos must be nonempty")
    d = rhos[0].shape[0]
    if any(rho.shape != (d, d) for rho in rhos):
        raise ValueError("all states must have the same dimension")
    n = d.bit_length() - 1
    expectations = [{} for _ in rhos]
    for u in range(4**n):
        matrix = pauli_matrix(unsigned_to_pauli_string(n, u))
        for coefficients, rho in zip(expectations, rhos):
            coefficients[u] = float(np.trace(matrix @ rho).real)
    return Ensemble(n, [1.0 / len(rhos)] * len(rhos) if priors is None else priors,
                    expectations, [str(i) for i in range(len(rhos))] if labels is None else labels,
                    validation_tolerance=tol)


def ensemble_to_density_matrices(ensemble):
    """Reconstruct dense matrices numerically; exact coefficients become floats."""
    matrices = []
    for coefficients in ensemble.expectations:
        rho = np.zeros((2**ensemble.n, 2**ensemble.n), dtype=complex)
        for u, coefficient in coefficients.items():
            rho += float(coefficient) * pauli_matrix(unsigned_to_pauli_string(ensemble.n, u))
        matrices.append(rho / 2**ensemble.n)
    return tuple(matrices)
