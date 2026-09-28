"""Bellman optimization over adaptive, commuting Pauli measurement trees."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Any, Mapping

from .ensemble import Ensemble, nonnegative_integer
from .pauli import canonical_key, code_from_key, extend_code, logical_representatives, unsigned_to_pauli_string
from .progress import ProgressUpdate


class SolverLimitError(RuntimeError):
    """Search exceeded its explicit budget; no optimum is being returned."""


@dataclass(frozen=True)
class SolveResult:
    success_probability: Any
    solved_projectors: int
    arithmetic: str
    best_actions: Mapping | None = None

    @property
    def probability(self):
        return float(self.success_probability)


class StabilizerDiscriminationSolver:
    """Optimize the supplied ensemble without symmetry assumptions.

    Exact input uses Q(sqrt(2)); floating input gives a numerical result.
    Memoization belongs to this instance and is released with it. The number
    of signed stabilizer projectors grows rapidly; use native benchmark
    backends for large cases. ``max_projectors`` aborts rather than silently
    returning a partially optimized value.
    """

    def __init__(self, ensemble: Ensemble, *, record_policy=False, max_projectors=None,
                 progress=None, progress_interval=0.2):
        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble must be an Ensemble")
        if max_projectors is not None:
            max_projectors = nonnegative_integer(max_projectors, "max_projectors")
            if not max_projectors:
                raise ValueError("max_projectors must be positive")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be a callable accepting a ProgressUpdate")
        if (isinstance(progress_interval, bool) or not isinstance(progress_interval, (int, float))
                or not math.isfinite(progress_interval) or progress_interval <= 0):
            raise ValueError("progress_interval must be finite and positive")
        self._ensemble = ensemble
        self.max_projectors = max_projectors
        self._values = {}
        self.actions = {} if record_policy else None
        self._progress = progress
        self._progress_interval = progress_interval
        self._progress_last = 0.0
        self._root_best = max(ensemble.priors)

    def _report_progress(self, *, force=False, phase="searching"):
        if self._progress is None:
            return
        now = time.monotonic()
        if not force and now - self._progress_last < self._progress_interval:
            return
        self._progress_last = now
        self._progress(ProgressUpdate(
            phase=phase, backend="python", solved_projectors=len(self._values),
            best_probability=float(self._root_best),
        ))

    @property
    def e(self):
        """The immutable input ensemble; construct a new solver to change it."""
        return self._ensemble

    @property
    def n(self):
        return self._ensemble.n

    def branch_scores(self, key):
        """Joint label/branch probabilities for a canonical signed code key."""
        code = code_from_key(key, self.n)
        mask = (1 << (2 * self.n)) - 1
        return tuple(prior * sum((-1 if p >> (2 * self.n) else 1) * coefficients.get(p & mask, 0)
                                 for p in code) / (1 << len(key))
                     for prior, coefficients in zip(self.e.priors, self.e.expectations))

    def _children(self, key, q):
        code = code_from_key(key, self.n)
        return tuple(canonical_key(extend_code(code, q, sign, self.n), self.n) for sign in (0, 1))

    def value(self, key):
        if key in self._values:
            return self._values[key]
        if self.max_projectors is not None and len(self._values) >= self.max_projectors:
            raise SolverLimitError(f"search reached max_projectors={self.max_projectors}")
        scores = self.branch_scores(key)
        best, mass = max(scores), sum(scores)
        action = ("guess", scores.index(best))
        if len(key) < self.n and best != mass:
            for q in logical_representatives(key, self.n):
                plus, minus = self._children(key, q)
                candidate = self.value(plus) + self.value(minus)
                if candidate > best:
                    best, action = candidate, ("measure", q)
                if not key:
                    self._root_best = best
                # Exact equality is a proof of optimality at this branch.
                if self.e.arithmetic == "exact" and best == mass:
                    break
        if self.max_projectors is not None and len(self._values) >= self.max_projectors:
            raise SolverLimitError(f"search reached max_projectors={self.max_projectors}")
        self._values[key] = best
        if self.actions is not None:
            self.actions[key] = action
        # Limit timing/callback overhead while still updating long searches.
        if self._progress is not None and len(self._values) % 128 == 0:
            self._report_progress()
        return best

    def solve(self):
        if () not in self._values:
            self._report_progress(force=True)
            try:
                value = self.value(())
            except BaseException as error:
                self._report_progress(force=True,
                                      phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
                raise
            self._root_best = value
            self._report_progress(force=True, phase="complete")
        else:
            value = self._values[()]
        actions = None if self.actions is None else MappingProxyType(dict(self.actions))
        return SolveResult(value, len(self._values), self.e.arithmetic, actions)

    def policy_tree(self):
        """Return the chosen decision tree as a JSON-serializable dictionary."""
        if self.actions is None:
            raise ValueError("construct with record_policy=True to export a policy")
        self.solve()

        def visit(key):
            kind, value = self.actions[key]
            if kind == "guess":
                return {"guess": self.e.labels[value], "label_index": value}
            plus, minus = self._children(key, value)
            return {"measure": unsigned_to_pauli_string(self.n, value),
                    "+1": visit(plus), "-1": visit(minus)}

        return visit(())


def success_probability(ensemble, **solver_options):
    """Return the optimal success value; use the solver for policy/statistics."""
    return StabilizerDiscriminationSolver(ensemble, **solver_options).solve().success_probability
