"""Exact benchmark ensembles and composable resource/copy constructors.

Pauli words and ket strings use the same convention: their first character is
qubit 0. Internally, qubit 0 occupies the least significant x/z mask bit.
All benchmark priors and expectations belong to ``Q(sqrt(2))``.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from operator import index

from .algebra import Qsqrt2
from .ensemble import Ensemble


def _count(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tensor_expectations(left, left_n: int, right, right_n: int):
    """Tensor sparse expectation maps, placing the left register first."""
    total_n = left_n + right_n
    left_mask = (1 << left_n) - 1
    right_mask = (1 << right_n) - 1
    result = {}
    for u, a in left.items():
        for v, b in right.items():
            x = (u & left_mask) | ((v & right_mask) << left_n)
            z = (u >> left_n) | ((v >> right_n) << left_n)
            coefficient = a * b
            if coefficient != 0:
                result[x | (z << total_n)] = coefficient
    return result


def _repeat_expectations(expectations, n: int, copies: int, *, exact: bool = True):
    result = {0: Qsqrt2(1) if exact else 1.0}
    if not exact:
        expectations = {u: float(c) for u, c in expectations.items()}
    width = 0
    for _ in range(copies):
        result = _tensor_expectations(result, width, expectations, n)
        width += n
    return result


def tensor_ensembles(*ensembles: Ensemble) -> Ensemble:
    """Discriminate all independent labels of a tensor-product ensemble.

    Each output hypothesis has a product prior. Registers and labels appear in
    argument order; the first input register contains qubit 0.
    """
    if not ensembles:
        raise ValueError("at least one ensemble is required")
    exact = all(e.arithmetic == "exact" for e in ensembles)
    cast = (lambda value: value) if exact else float
    n = sum(ensemble.n for ensemble in ensembles)
    states, priors, labels = [], [], []
    for choices in product(*(range(len(e.priors)) for e in ensembles)):
        state = {0: Qsqrt2(1) if exact else 1.0}
        prior = Qsqrt2(1) if exact else 1.0
        width = 0
        label = []
        for ensemble, choice in zip(ensembles, choices):
            state = _tensor_expectations(
                state, width,
                {u: cast(c) for u, c in ensemble.expectations[choice].items()},
                ensemble.n,
            )
            width += ensemble.n
            prior *= cast(ensemble.priors[choice])
            label.append(str(ensemble.labels[choice]))
        states.append(state)
        priors.append(prior)
        labels.append(repr(tuple(label)))
    return Ensemble(n, priors, states, labels,
                    validation_tolerance=sum(e.validation_tolerance for e in ensembles))


def append_t_states(ensemble: Ensemble, tstates: int = 0) -> Ensemble:
    r"""Append common copies of ``|T>=(|0>+exp(i*pi/4)|1>)/sqrt(2)``.

    Resource qubits follow all system qubits. They do not introduce additional
    labels or change the prior distribution.
    """
    tstates = _count(tstates, "tstates")
    if tstates == 0:
        return ensemble
    inv_sqrt2 = Qsqrt2(0, Fraction(1, 2))
    resource = _repeat_expectations(
        {0: Qsqrt2(1), 1: inv_sqrt2, 3: inv_sqrt2}, 1, tstates,
        exact=ensemble.arithmetic == "exact",
    )
    states = [
        _tensor_expectations(state, ensemble.n, resource, tstates)
        for state in ensemble.expectations
    ]
    return Ensemble(ensemble.n + tstates, ensemble.priors, states, ensemble.labels,
                    validation_tolerance=ensemble.validation_tolerance)


def xor_repetition(ensemble: Ensemble, copies: int = 2) -> Ensemble:
    r"""Discriminate the parity of independent, equally likely binary labels.

    For ``A=(rho0+rho1)/2`` and ``D=(rho0-rho1)/2``, the two conditional
    hypotheses are ``A**tensor(copies) +/- D**tensor(copies)``, each with prior
    one half. Input labels 0 and 1 mean their positions in the ensemble.
    """
    copies = _count(copies, "copies", minimum=1)
    exact = ensemble.arithmetic == "exact"
    half = Qsqrt2(Fraction(1, 2)) if exact else 0.5
    zero = Qsqrt2() if exact else 0.0
    if len(ensemble.priors) != 2 or any(p != half for p in ensemble.priors):
        raise ValueError("XOR repetition requires two equally likely hypotheses")
    first, second = ensemble.expectations
    mean, difference = {}, {}
    for u in first.keys() | second.keys():
        a, b = first.get(u, zero), second.get(u, zero)
        average, delta = (a + b) / 2, (a - b) / 2
        if average != 0:
            mean[u] = average
        if delta != 0:
            difference[u] = delta
    mean = _repeat_expectations(mean, ensemble.n, copies, exact=exact)
    difference = _repeat_expectations(difference, ensemble.n, copies, exact=exact)
    states = []
    for sign in (1, -1):
        state = {}
        for u in mean.keys() | difference.keys():
            coefficient = mean.get(u, zero) + sign * difference.get(u, zero)
            if coefficient != 0:
                state[u] = coefficient
        states.append(state)
    return Ensemble(ensemble.n * copies, [half, half], states, ["even", "odd"],
                    validation_tolerance=copies * ensemble.validation_tolerance)


def _uniform(n: int, states, labels) -> Ensemble:
    return Ensemble(
        n, [Qsqrt2(Fraction(1, len(states)))] * len(states), states, labels
    )


def _product_state(ket: str):
    """Exact Pauli expectations for a product of 0, 1, +, and - states."""
    axes = {"0": (0, 1, 1), "1": (0, 1, -1),
            "+": (1, 0, 1), "-": (1, 0, -1)}
    result = {}
    for selected in product((False, True), repeat=len(ket)):
        x = z = 0
        coefficient = 1
        for qubit, (character, use_axis) in enumerate(zip(ket, selected)):
            if use_axis:
                xb, zb, sign = axes[character]
                x |= xb << qubit
                z |= zb << qubit
                coefficient *= sign
        result[x | (z << len(ket))] = Qsqrt2(coefficient)
    return result


def _cat_state(first: str, second: str, sign: int):
    """Pauli expectations of (|first> + sign |second>)/sqrt(2)."""
    n = len(first)
    u = sum(int(bit) << q for q, bit in enumerate(first))
    v = sum(int(bit) << q for q, bit in enumerate(second))
    d = u ^ v
    result = {}
    for x in (0, d):
        for z in range(1 << n):
            if (z & d).bit_count() % 2:
                continue
            phase = (-1) ** ((z & u).bit_count() + (x & z).bit_count() // 2)
            result[x | (z << n)] = Qsqrt2(phase * (sign if x else 1))
    return result


def _mixture(states):
    result = {}
    weight = Fraction(1, len(states))
    for state in states:
        for u, coefficient in state.items():
            result[u] = result.get(u, Qsqrt2()) + coefficient * weight
    return {u: coefficient for u, coefficient in result.items() if coefficient != 0}


def e8_full(tstates: int = 0) -> Ensemble:
    """Uniform full-label ensemble of the eight E8 basis states.

    The pairs are (000,111), (010,110), (001,011), and (100,101),
    with the plus state preceding the minus state within each pair.
    """
    states, labels = [], []
    for pair, (first, second) in enumerate(
        [("000", "111"), ("010", "110"), ("001", "011"), ("100", "101")]
    ):
        for sign in (1, -1):
            states.append(_cat_state(first, second, sign))
            labels.append(f"{pair}{'+' if sign == 1 else '-'}")
    return append_t_states(_uniform(3, states, labels), tstates)


def e8_sign(tstates: int = 0) -> Ensemble:
    """Binary E8 sign mixtures, uniform over the four pair labels."""
    full = e8_full()
    states = [_mixture(full.expectations[sign::2]) for sign in range(2)]
    return append_t_states(_uniform(3, states, ["+", "-"]), tstates)


def e6_full(tstates: int = 0) -> Ensemble:
    """Uniform six-label ensemble |±10>, |0±1>, and |10±>."""
    kets = ["+10", "-10", "0+1", "0-1", "10+", "10-"]
    states = [_product_state(ket) for ket in kets]
    labels = ["A+", "A-", "B+", "B-", "C+", "C-"]
    return append_t_states(_uniform(3, states, labels), tstates)


def e6_sign(tstates: int = 0) -> Ensemble:
    """Binary mixtures of the three positive/negative E6 sign states."""
    full = e6_full()
    states = [_mixture(full.expectations[sign::2]) for sign in range(2)]
    return append_t_states(_uniform(3, states, ["+", "-"]), tstates)


def twoqubit_mixture(tstates: int = 0) -> Ensemble:
    """Equiprobable (|0+><0+|+|+0><+0|)/2 and (|11><11|+|--><--|)/2."""
    states = [
        _mixture([_product_state("0+"), _product_state("+0")]),
        _mixture([_product_state("11"), _product_state("--")]),
    ]
    return append_t_states(_uniform(2, states, ["0", "1"]), tstates)


def four_state(tstates: int = 0) -> Ensemble:
    """Uniform four-label ensemble |+10>, |0+1>, |10+>, |--->."""
    kets = ["+10", "0+1", "10+", "---"]
    return append_t_states(
        _uniform(3, [_product_state(ket) for ket in kets], kets), tstates
    )
