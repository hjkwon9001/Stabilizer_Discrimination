"""State definitions checked against independent dense constructions."""

from fractions import Fraction
from itertools import product
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from stabdisc.ensemble import Ensemble
from stabdisc.numeric import density_matrices_to_ensemble, ensemble_to_density_matrices
from stabdisc.states import (
    append_t_states, e6_full, e6_sign, e8_full, e8_sign, four_state,
    tensor_ensembles, twoqubit_mixture, xor_repetition,
)


def projector(vector):
    return np.outer(vector, vector.conj())


def product_ket(word):
    table = {"0": np.array([1, 0]), "1": np.array([0, 1]),
             "+": np.array([1, 1]) / np.sqrt(2),
             "-": np.array([1, -1]) / np.sqrt(2)}
    state = np.array([1.0])
    for character in word:
        state = np.kron(state, table[character])
    return state


@unittest.skipIf(np is None, "NumPy is required for dense state validation")
class DenseStateTests(unittest.TestCase):
    def assert_density_matches(self, ensemble, expected):
        actual = ensemble_to_density_matrices(ensemble)
        self.assertEqual(len(actual), len(expected))
        for obtained, wanted in zip(actual, expected):
            np.testing.assert_allclose(obtained, wanted, atol=1e-12, rtol=0)

    def test_e8_ket_convention_and_sign_sectors(self):
        expected = []
        for first, second in [("000", "111"), ("010", "110"),
                              ("001", "011"), ("100", "101")]:
            for sign in (1, -1):
                vector = np.zeros(8, dtype=complex)
                vector[int(first, 2)] = 1 / np.sqrt(2)
                vector[int(second, 2)] = sign / np.sqrt(2)
                expected.append(projector(vector))
        self.assert_density_matches(e8_full(), expected)
        self.assert_density_matches(e8_sign(), [sum(expected[::2]) / 4, sum(expected[1::2]) / 4])

    def test_e6_and_four_state_kets(self):
        expected = [projector(product_ket(word)) for word in
                    ["+10", "-10", "0+1", "0-1", "10+", "10-"]]
        self.assert_density_matches(e6_full(), expected)
        self.assert_density_matches(e6_sign(), [sum(expected[::2]) / 3, sum(expected[1::2]) / 3])
        self.assert_density_matches(four_state(), [projector(product_ket(word))
                                             for word in ["+10", "0+1", "10+", "---"]])

    def test_twoqubit_mixture_definition(self):
        expected = [(projector(product_ket(a)) + projector(product_ket(b))) / 2
                    for a, b in [("0+", "+0"), ("11", "--")]]
        self.assert_density_matches(twoqubit_mixture(), expected)

    def test_builtins_are_physical_and_exact(self):
        for constructor in [e8_full, e8_sign, e6_full, e6_sign, four_state, twoqubit_mixture]:
            for resources in (0, 1):
                with self.subTest(constructor=constructor.__name__, resources=resources):
                    ensemble = constructor(tstates=resources)
                    self.assertEqual(ensemble.arithmetic, "exact")
                    self.assertIs(ensemble.validate_physical(), ensemble)

    def test_common_t_state_dense_tensor_order(self):
        for floating in (False, True):
            with self.subTest(floating=floating):
                base = twoqubit_mixture()
                matrices = ensemble_to_density_matrices(base)
                if floating:
                    base = density_matrices_to_ensemble(matrices, labels=base.labels)
                resource = projector(np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2))
                actual = append_t_states(base, 2)
                expected = [np.kron(np.kron(rho, resource), resource) for rho in matrices]
                self.assert_density_matches(actual, expected)
                self.assertEqual(actual.labels, base.labels)
                self.assertEqual(actual.priors, base.priors)
                self.assertEqual(actual.arithmetic, "floating" if floating else "exact")

    def test_xor_states_match_conditional_parity_mixtures(self):
        for floating, copies in product((False, True), (1, 2)):
            with self.subTest(floating=floating, copies=copies):
                base = twoqubit_mixture()
                matrices = ensemble_to_density_matrices(base)
                if floating:
                    base = density_matrices_to_ensemble(matrices)
                expected = [np.zeros((4**copies, 4**copies), dtype=complex) for _ in range(2)]
                for bits in product((0, 1), repeat=copies):
                    state = np.array([[1.0]])
                    for bit in bits:
                        state = np.kron(state, matrices[bit])
                    expected[sum(bits) % 2] += state / 2 ** (copies - 1)
                actual = xor_repetition(base, copies)
                self.assert_density_matches(actual, expected)
                self.assertEqual(actual.arithmetic, "floating" if floating else "exact")

    def test_tensor_ensembles_handles_mixed_arithmetic_and_product_priors(self):
        exact = Ensemble.from_pauli_dicts(
            1, [{"I": 1, "Z": 1}, {"I": 1, "Z": -1}],
            priors=[Fraction(1, 3), Fraction(2, 3)], labels=["a ⊗ b", "a"],
        )
        floating = density_matrices_to_ensemble(
            [projector(product_ket("+")), projector(product_ket("-"))],
            priors=[0.25, 0.75], labels=["c", "b ⊗ c"],
        )
        actual = tensor_ensembles(exact, floating)
        expected = [np.kron(left, right)
                    for left in ensemble_to_density_matrices(exact)
                    for right in ensemble_to_density_matrices(floating)]
        self.assert_density_matches(actual, expected)
        self.assertEqual(actual.arithmetic, "floating")
        np.testing.assert_allclose(actual.priors, [1 / 12, 1 / 4, 1 / 6, 1 / 2])
        self.assertEqual(len(set(actual.labels)), 4)
        self.assertEqual(tensor_ensembles(exact, exact).arithmetic, "exact")


class StateValidationTests(unittest.TestCase):
    def test_xor_requires_balanced_binary_input(self):
        with self.assertRaisesRegex(ValueError, "equally likely"):
            xor_repetition(e8_full())
        nonuniform = Ensemble(2, [Fraction(1, 3), Fraction(2, 3)],
                              twoqubit_mixture().expectations, ["0", "1"])
        with self.assertRaisesRegex(ValueError, "equally likely"):
            xor_repetition(nonuniform)

    def test_resource_and_copy_count_validation(self):
        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    append_t_states(e8_sign(), invalid)
                with self.assertRaises(ValueError):
                    xor_repetition(e8_sign(), invalid)
                with self.assertRaises(ValueError):
                    e6_full(tstates=invalid)

    def test_empty_tensor_and_zero_copies_are_rejected(self):
        with self.assertRaises(ValueError):
            tensor_ensembles()
        with self.assertRaises(ValueError):
            xor_repetition(twoqubit_mixture(), 0)


if __name__ == "__main__":
    unittest.main()
