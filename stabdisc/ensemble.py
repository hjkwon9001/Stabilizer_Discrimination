"""Immutable finite ensembles represented by sparse Pauli expectations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from numbers import Integral, Rational, Real
from types import MappingProxyType
from typing import Mapping, Sequence

from .algebra import Qsqrt2
from .pauli import pauli_string_to_unsigned


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class Ensemble:
    """A normalized ensemble of ``n``-qubit states.

    ``expectations[i][u]`` is Tr(P_u rho_i); absent terms mean zero.
    The identity (key 0) must explicitly have expectation one. Integers,
    fractions and Qsqrt2 values are converted to exact Qsqrt2 arithmetic;
    any floating input selects floating arithmetic for the entire ensemble.

    Construction validates normalization, finiteness and coefficient bounds.
    These checks do not prove positivity. Call ``validate_physical`` for a
    dense numerical PSD check when supplying untrusted Pauli coefficients.
    """

    n: int
    priors: Sequence
    expectations: Sequence[Mapping]
    labels: Sequence[str]
    validation_tolerance: float = 1e-10
    arithmetic: str = field(init=False)

    def __post_init__(self):
        n = nonnegative_integer(self.n, "n")
        if not isfinite(self.validation_tolerance) or self.validation_tolerance < 0:
            raise ValueError("validation_tolerance must be finite and nonnegative")
        priors, states, labels = tuple(self.priors), tuple(self.expectations), tuple(self.labels)
        if not priors or len(priors) != len(states) or len(priors) != len(labels):
            raise ValueError("a nonempty ensemble needs equal numbers of priors, states and labels")
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique strings")
        if any(not isinstance(state, Mapping) for state in states):
            raise ValueError("each state must be a mapping of Pauli indices to expectations")
        values = priors + tuple(c for state in states for c in state.values())
        if any(not isinstance(v, (Real, Qsqrt2)) for v in values):
            raise ValueError("coefficients must be real numbers or Qsqrt2")
        exact = all(isinstance(v, (Rational, Qsqrt2)) for v in values)
        cast = (lambda v: v if isinstance(v, Qsqrt2) else Qsqrt2(Fraction(v))) if exact else float
        priors = tuple(cast(p) for p in priors)
        tolerance = 0 if exact else self.validation_tolerance
        if not exact and any(not isfinite(float(v)) for v in values):
            raise ValueError("coefficients must be finite")
        if any(p < 0 for p in priors):
            raise ValueError("priors must be nonnegative")
        normalized = sum(priors) == 1 if exact else abs(sum(priors) - 1) <= tolerance
        if not normalized:
            raise ValueError("priors must sum to one")
        copied = []
        for state in states:
            if any(isinstance(u, bool) or not isinstance(u, Integral) or not 0 <= u < 4**n for u in state):
                raise ValueError("Pauli indices must be integers in [0, 4**n)")
            coefficients = {int(u): cast(c) for u, c in state.items()}
            identity = coefficients.get(0)
            if identity is None or ((identity != 1) if exact else abs(identity - 1) > tolerance):
                raise ValueError("every state must explicitly include identity expectation one")
            bound = Qsqrt2(1) if exact else 1 + tolerance
            if any(abs(c) > bound for c in coefficients.values()):
                raise ValueError("Pauli expectations must lie in [-1, 1]")
            copied.append(MappingProxyType(coefficients))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "expectations", tuple(copied))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "arithmetic", "exact" if exact else "floating")

    @classmethod
    def from_pauli_dicts(cls, n, states, priors=None, labels=None, *, check_physical=True, tol=1e-10):
        """Construct from e.g. ``{'II': 1, 'XI': Fraction(1, 2)}``.

        Character 0 is qubit 0, the leftmost tensor factor in dense matrices.
        A dense numerical PSD check runs by default; disable it only for
        trusted symbolic inputs. Its cost grows exponentially with n.
        """
        n = nonnegative_integer(n, "n")
        states = tuple(states)
        if not states:
            raise ValueError("states must be nonempty")
        encoded = []
        for state in states:
            converted = {}
            for label, value in state.items():
                if not isinstance(label, str) or len(label) != n or any(c not in "IXYZ" for c in label.upper()):
                    raise ValueError(f"Pauli labels must contain exactly {n} I/X/Y/Z characters")
                index = pauli_string_to_unsigned(label)
                if index in converted:
                    raise ValueError("duplicate Pauli after case normalization")
                converted[index] = value
            encoded.append(converted)
        ensemble = cls(n, [Fraction(1, len(states))] * len(states) if priors is None else priors,
                       encoded, [str(i) for i in range(len(states))] if labels is None else labels,
                       validation_tolerance=tol)
        if check_physical:
            ensemble.validate_physical(tol=tol)
        return ensemble

    def validate_physical(self, *, tol=None):
        """Numerically check PSD of reconstructed density matrices; return self."""
        from .numeric import ensemble_to_density_matrices, validate_density_matrix
        tol = self.validation_tolerance if tol is None else tol
        for rho in ensemble_to_density_matrices(self):
            validate_density_matrix(rho, tol=tol)
        return self
