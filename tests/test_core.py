"""Independent algebra, physics, input-validation, and solver regressions.

Run with ``python -m unittest discover -s tests -v`` after installation.
The matrix reference below deliberately does not use the package's Pauli
enumeration, signed-code representation, or Bellman implementation.
"""

from fractions import Fraction
from itertools import product
import json
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from stabdisc import (
    Ensemble,
    Qsqrt2,
    StabilizerDiscriminationSolver,
    e6_full,
    e6_sign,
    e8_full,
    e8_sign,
    four_state,
    success_probability,
    twoqubit_mixture,
)
from stabdisc.pauli import (
    canonical_key,
    code_from_key,
    commute_unsigned,
    extend_code,
    logical_representatives,
    multiply,
    pauli_string_to_unsigned,
)


def binary_z_ensemble(priors=None):
    return Ensemble.from_pauli_dicts(
        1,
        [{"I": 1, "Z": 1}, {"I": 1, "Z": -1}],
        priors=priors,
        labels=["zero", "one"],
    )


def matrix_pauli(n, unsigned):
    """Decode the documented bit convention without library helpers."""
    factors = (
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
    )
    result = np.ones((1, 1), dtype=complex)
    for i in range(n):
        x = (unsigned >> i) & 1
        z = (unsigned >> (n + i)) & 1
        result = np.kron(result, factors[x + 2 * z])
    return result


def dense_reference_optimum(rhos, priors):
    """Exhaust all commuting projective refinements using dense matrices."""
    d = rhos[0].shape[0]
    n = d.bit_length() - 1
    single = (
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
    )
    paulis = []
    for letters in product(range(4), repeat=n):
        if not any(letters):
            continue
        p = np.ones((1, 1), dtype=complex)
        for letter in letters:
            p = np.kron(p, single[letter])
        paulis.append(p)
    memo = {}

    def value(projector):
        key = tuple(projector.real.ravel()) + tuple(projector.imag.ravel())
        if key in memo:
            return memo[key]
        best = max(
            prior * float(np.trace(projector @ rho).real)
            for prior, rho in zip(priors, rhos)
        )
        if np.isclose(np.trace(projector).real, 1):
            memo[key] = best
            return best
        for p in paulis:
            if not np.allclose(projector @ p, p @ projector, atol=1e-14):
                continue
            plus = (projector + projector @ p) / 2
            minus = (projector - projector @ p) / 2
            if np.allclose(plus, 0) or np.allclose(minus, 0):
                continue
            best = max(best, value(plus) + value(minus))
        memo[key] = best
        return best

    return value(np.eye(d, dtype=complex))


class ExactAlgebraTests(unittest.TestCase):
    def test_rational_interoperability(self):
        self.assertEqual(Qsqrt2(Fraction(3, 4)), Fraction(3, 4))
        self.assertEqual(Qsqrt2(1), 1)
        self.assertEqual(hash(Qsqrt2(1)), hash(1))
        self.assertEqual(Qsqrt2(1, 1) * Qsqrt2(1, -1), -1)
        self.assertEqual(2 / Qsqrt2(0, 1), Qsqrt2(0, 1))
        self.assertEqual(3 - Qsqrt2(1, 2), Qsqrt2(2, -2))

    def test_ordering_does_not_use_float_conversion(self):
        # Consecutive Pell solutions approach sqrt(2) much more closely than
        # double precision can resolve. Their exact signs alternate.
        p, q = 1, 1
        for _ in range(80):
            p, q = p + 2 * q, p + q
        for _ in range(2):
            difference = Qsqrt2(p, -q)
            self.assertEqual(difference > 0, p * p > 2 * q * q)
            self.assertEqual(difference < 0, p * p < 2 * q * q)
            self.assertNotEqual(difference, 0)
            p, q = p + 2 * q, p + q

    def test_division_inverse_and_zero(self):
        for a, b in [(1, 2), (-3, 1), (0, Fraction(1, 2))]:
            value = Qsqrt2(a, b)
            self.assertEqual(value / value, 1)
            self.assertEqual((1 / value) * value, 1)
        with self.assertRaises(ZeroDivisionError):
            Qsqrt2(1) / 0


class PauliEnumerationTests(unittest.TestCase):
    def test_explicit_bit_and_tensor_order(self):
        self.assertEqual(pauli_string_to_unsigned("XZ"), 9)
        self.assertEqual(pauli_string_to_unsigned("YI"), 5)

    def test_signed_projector_counts_and_roundtrip(self):
        expected = {1: [1, 6], 2: [1, 30, 60], 3: [1, 126, 1260, 1080]}
        for n, counts in expected.items():
            with self.subTest(n=n):
                layer = {()}
                actual = [1]
                for rank in range(n):
                    next_layer = set()
                    for key in layer:
                        code = code_from_key(key, n)
                        self.assertEqual(canonical_key(tuple(reversed(code)), n), key)
                        reps = logical_representatives(key, n)
                        self.assertEqual(len(reps), 4 ** (n - rank) - 1)
                        for q in reps:
                            for sign in (0, 1):
                                child = canonical_key(extend_code(code, q, sign, n), n)
                                self.assertEqual(len(child), rank + 1)
                                next_layer.add(child)
                    layer = next_layer
                    actual.append(len(layer))
                self.assertEqual(actual, counts)


class EnsembleAndSolverTests(unittest.TestCase):
    def test_solver_input_cannot_be_reassigned_after_caching(self):
        solver = StabilizerDiscriminationSolver(binary_z_ensemble())
        solver.solve()
        with self.assertRaises(AttributeError):
            solver.e = binary_z_ensemble()
        with self.assertRaises(AttributeError):
            solver.n = 2

    def test_default_priors_remain_exact(self):
        value = success_probability(binary_z_ensemble())
        self.assertEqual(value, 1)
        self.assertNotIsInstance(value, float)

    def test_identical_hypotheses_and_biased_priors(self):
        ensemble = Ensemble.from_pauli_dicts(
            1, [{"I": 1}, {"I": 1}], [Fraction(1, 3), Fraction(2, 3)]
        )
        self.assertEqual(success_probability(ensemble), Fraction(2, 3))

    def test_one_qubit_formula_with_unequal_priors(self):
        p, q = Fraction(2, 5), Fraction(3, 5)
        r0 = [Fraction(1, 3), Fraction(1, 4), Fraction(-1, 5)]
        r1 = [Fraction(-1, 2), Fraction(1, 5), Fraction(1, 3)]
        states = [dict(zip("IXYZ", [1] + r)) for r in (r0, r1)]
        expected = max(
            max(p * (1 + a) / 2, q * (1 + b) / 2)
            + max(p * (1 - a) / 2, q * (1 - b) / 2)
            for a, b in zip(r0, r1)
        )
        self.assertEqual(success_probability(Ensemble.from_pauli_dicts(1, states, [p, q])), expected)

    def test_common_stabilizer_ancilla_preserves_value(self):
        base = Ensemble.from_pauli_dicts(1, [{"I": 1, "Z": 1}, {"I": 1, "X": 1}])
        assisted = Ensemble.from_pauli_dicts(
            2,
            [
                {"II": 1, "ZI": 1, "IZ": 1, "ZZ": 1},
                {"II": 1, "XI": 1, "IZ": 1, "XZ": 1},
            ],
        )
        self.assertEqual(success_probability(base), Fraction(3, 4))
        self.assertEqual(success_probability(assisted), success_probability(base))

    def test_input_is_copied_and_immutable(self):
        priors = [Fraction(1, 2), Fraction(1, 2)]
        states = [{0: 1, 2: 1}, {0: 1, 2: -1}]
        labels = ["zero", "one"]
        ensemble = Ensemble(1, priors, states, labels)
        states[0][2] = -1
        priors[0] = 0
        labels[0] = "changed"
        self.assertEqual(ensemble.expectations[0][2], 1)
        self.assertEqual(ensemble.priors[0], Fraction(1, 2))
        self.assertEqual(ensemble.labels[0], "zero")
        with self.assertRaises(TypeError):
            ensemble.expectations[0][2] = 0
        self.assertEqual(success_probability(ensemble), 1)

    def test_instances_have_independent_memo_tables(self):
        left = StabilizerDiscriminationSolver(binary_z_ensemble())
        right = StabilizerDiscriminationSolver(binary_z_ensemble())
        left.solve()
        self.assertIsNot(left._values, right._values)
        self.assertTrue(left._values)
        self.assertFalse(right._values)
        again = left.solve()
        self.assertEqual(again.success_probability, 1)
        self.assertEqual(again.solved_projectors, len(left._values))

    def test_search_budget_raises_without_returning_partial_optimum(self):
        from stabdisc.solver import SolverLimitError

        solver = StabilizerDiscriminationSolver(e8_sign(), max_projectors=3)
        with self.assertRaises(SolverLimitError):
            solver.solve()
        self.assertLessEqual(len(solver._values), 3)

    def test_rejects_invalid_ensemble_input(self):
        cases = [
            lambda: Ensemble(1, [], [], []),
            lambda: binary_z_ensemble([1, 1]),
            lambda: binary_z_ensemble([Fraction(-1, 2), Fraction(3, 2)]),
            lambda: Ensemble.from_pauli_dicts(2, [{"I": 1}], [1]),
            lambda: Ensemble.from_pauli_dicts(1, [{"I": 1, "Z": 2}], [1]),
            lambda: Ensemble.from_pauli_dicts(1, [{"Z": 1}], [1]),
            lambda: Ensemble.from_pauli_dicts(1, [{"I": float("nan")}], [1]),
            lambda: Ensemble.from_pauli_dicts(1, [{"I": 1}], [float("nan")]),
            lambda: Ensemble(1, [1], [{0: 1, 4: 1}], ["invalid-index"]),
        ]
        for index, constructor in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    constructor()

    def test_known_small_exact_examples(self):
        cases = [
            (e8_full, Fraction(3, 4)),
            (e8_sign, Fraction(3, 4)),
            (e6_full, Fraction(5, 6)),
            (e6_sign, Fraction(5, 6)),
            (twoqubit_mixture, Fraction(7, 8)),
            (four_state, Fraction(15, 16)),
        ]
        for constructor, expected in cases:
            with self.subTest(example=constructor.__name__):
                self.assertEqual(success_probability(constructor()), expected)

    def test_nonrational_three_qubit_example(self):
        self.assertEqual(
            success_probability(twoqubit_mixture(tstates=1)),
            Qsqrt2(Fraction(3, 4), Fraction(1, 8)),
        )


@unittest.skipIf(np is None, "NumPy is required for independent matrix tests")
class DenseMatrixTests(unittest.TestCase):
    def test_density_validation_tolerance_is_consistent(self):
        from stabdisc import density_matrices_to_ensemble, append_t_states
        rho = np.diag([1 + 5e-9, -5e-9])
        with self.assertRaises(ValueError):
            density_matrices_to_ensemble([rho])
        ensemble = density_matrices_to_ensemble([rho], tol=1e-6)
        self.assertEqual(ensemble.validation_tolerance, 1e-6)
        self.assertEqual(append_t_states(ensemble, 1).validation_tolerance, 1e-6)
        ensemble.validate_physical()

    def test_exported_policy_replays_to_the_reported_probability(self):
        ensemble = four_state()
        solver = StabilizerDiscriminationSolver(ensemble, record_policy=True)
        result = solver.solve()
        tree = solver.policy_tree()
        json.dumps(tree)  # The public export is a serializable experiment.
        dimension = 2 ** ensemble.n
        rhos = []
        for expectations in ensemble.expectations:
            rho = np.zeros((dimension, dimension), dtype=complex)
            for unsigned, coefficient in expectations.items():
                rho += float(coefficient) * matrix_pauli(ensemble.n, unsigned)
            rhos.append(rho / dimension)
        factors = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.diag([1, -1]).astype(complex),
        }

        def replay(node, branches, depth=0):
            self.assertLessEqual(depth, ensemble.n)
            if "guess" in node:
                label = node["label_index"]
                self.assertEqual(node["guess"], ensemble.labels[label])
                return float(ensemble.priors[label]) * np.trace(branches[label]).real
            pauli = np.ones((1, 1), dtype=complex)
            for character in node["measure"]:
                pauli = np.kron(pauli, factors[character])
            probability = 0
            for sign, outcome in [(1, "+1"), (-1, "-1")]:
                projector = (np.eye(dimension) + sign * pauli) / 2
                child = [projector @ rho @ projector for rho in branches]
                probability += replay(node[outcome], child, depth + 1)
            return probability

        self.assertAlmostEqual(replay(tree, rhos), float(result.success_probability), places=13)

    def test_benchmark_definitions_against_explicit_kets(self):
        def product_ket(word):
            factors = {
                "0": np.array([1, 0], dtype=complex),
                "1": np.array([0, 1], dtype=complex),
                "+": np.array([1, 1], dtype=complex) / np.sqrt(2),
                "-": np.array([1, -1], dtype=complex) / np.sqrt(2),
            }
            vector = np.ones(1, dtype=complex)
            for character in word:
                vector = np.kron(vector, factors[character])
            return vector

        def density(vector):
            return np.outer(vector, vector.conj())

        e8_rhos = []
        for first, second in [("000", "111"), ("010", "110"), ("001", "011"), ("100", "101")]:
            for sign in (1, -1):
                e8_rhos.append(density((product_ket(first) + sign * product_ket(second)) / np.sqrt(2)))
        e6_rhos = [density(product_ket(word)) for word in ["+10", "-10", "0+1", "0-1", "10+", "10-"]]
        four_rhos = [density(product_ket(word)) for word in ["+10", "0+1", "10+", "---"]]
        two_rhos = [
            (density(product_ket("0+")) + density(product_ket("+0"))) / 2,
            (density(product_ket("11")) + density(product_ket("--"))) / 2,
        ]
        for ensemble, rhos in [(e8_full(), e8_rhos), (e6_full(), e6_rhos), (four_state(), four_rhos), (twoqubit_mixture(), two_rhos)]:
            for expectations, rho in zip(ensemble.expectations, rhos):
                for unsigned in range(4 ** ensemble.n):
                    expected = np.trace(matrix_pauli(ensemble.n, unsigned) @ rho)
                    self.assertAlmostEqual(float(expectations.get(unsigned, 0)), expected.real, places=13)
                    self.assertAlmostEqual(expected.imag, 0, places=13)

    def test_two_qubit_pauli_multiplication_against_matrices(self):
        n = 2
        for u, v in product(range(4 ** n), repeat=2):
            left, right = matrix_pauli(n, u), matrix_pauli(n, v)
            expected_commute = np.allclose(left @ right, right @ left)
            self.assertEqual(commute_unsigned(n, u, v), expected_commute)
            if expected_commute:
                for s, t in product((0, 1), repeat=2):
                    encoded = multiply(n, u | (s << (2 * n)), v | (t << (2 * n)))
                    sign = (encoded >> (2 * n)) & 1
                    actual = (-1) ** sign * matrix_pauli(n, encoded & 15)
                    np.testing.assert_array_equal(actual, (-1) ** (s + t) * (left @ right))
            else:
                with self.assertRaises(ValueError):
                    multiply(n, u, v)

    def test_arbitrary_mixed_ensemble_against_dense_exhaustive_reference(self):
        from stabdisc.numeric import density_matrices_to_ensemble

        rng = np.random.default_rng(1409)
        rhos = []
        for _ in range(3):
            a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = a @ a.conj().T
            rhos.append(rho / np.trace(rho))
        priors = np.array([0.2, 0.3, 0.5])
        expected = dense_reference_optimum(rhos, priors)
        actual = success_probability(density_matrices_to_ensemble(rhos, priors=priors))
        self.assertAlmostEqual(float(actual), expected, places=12)

    def test_density_validation(self):
        from stabdisc.numeric import density_matrices_to_ensemble

        invalid = [
            [],
            [np.eye(2)],  # Trace two.
            [np.diag([1.1, -0.1])],
            [np.array([[1, 1], [0, 0]], dtype=complex)],
            [np.array([[float("nan"), 0], [0, 1]])],
            [np.ones((2, 3))],
            [np.eye(3) / 3],
        ]
        for index, rhos in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    density_matrices_to_ensemble(rhos)


if __name__ == "__main__":
    unittest.main()
