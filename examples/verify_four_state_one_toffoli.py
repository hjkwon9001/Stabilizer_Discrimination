#!/usr/bin/env python3
"""Verify perfect discrimination of the four-state Shifts UPB with one Toffoli.

States (qubits ordered 1,2,3):
    0: |+10>
    1: |0+1>
    2: |10+>
    3: |--->

Protocol:
  1. Apply the listed Clifford pre-processing circuit.
  2. Apply CCX with controls qubits 1,2 and target qubit 3.
  3. Apply the adaptive Pauli measurement tree printed below.

The script computes every branch probability and verifies unit success using
floating-point density amplitudes. It checks the supplied CCX circuit, not a
four-T-state injection circuit. Converting the CCX witness into a resource-state
count requires a specified measurement-assisted implementation and accounting
for its ancillary resources.
"""

from __future__ import annotations

import argparse
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)
plus = (zero + one) / np.sqrt(2)
minus = (zero - one) / np.sqrt(2)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


STATES = [
    kron3(plus, one, zero),
    kron3(zero, plus, one),
    kron3(one, zero, plus),
    kron3(minus, minus, minus),
]
LABELS = ["|+10>", "|0+1>", "|10+>", "|--->"]


def one_qubit_gate(gate, qubit):
    factors = [I, I, I]
    factors[qubit] = gate
    return kron3(*factors)


def cnot(control, target):
    out = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        bits = [(column >> (2 - q)) & 1 for q in range(3)]
        mapped = bits.copy()
        mapped[target] ^= bits[control]
        row = (mapped[0] << 2) | (mapped[1] << 1) | mapped[2]
        out[row, column] = 1
    return out


def toffoli(control1=0, control2=1, target=2):
    out = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        bits = [(column >> (2 - q)) & 1 for q in range(3)]
        mapped = bits.copy()
        mapped[target] ^= bits[control1] & bits[control2]
        row = (mapped[0] << 2) | (mapped[1] << 1) | mapped[2]
        out[row, column] = 1
    return out


# Chronological gate order; qubits are zero-indexed internally.
CLIFFORD_SEQUENCE = [
    ("CX", 2, 1),
    ("CX", 0, 2),
    ("CX", 2, 0),
    ("S", 2),
    ("CX", 0, 1),
    ("S", 1),
    ("CX", 0, 2),
    ("CX", 1, 2),
    ("Z", 1),
    ("CX", 2, 1),
    ("H", 0),
    ("S", 1),
    ("CX", 0, 2),
    ("H", 0),
    ("S", 0),
    ("S", 2),
    ("H", 2),
    ("CX", 1, 2),
    ("S", 0),
    ("CX", 0, 1),
    ("CX", 0, 2),
]


def preprocessing_unitary():
    unitary = np.eye(8, dtype=complex)
    for instruction in CLIFFORD_SEQUENCE:
        name = instruction[0]
        if name == "CX":
            gate = cnot(instruction[1], instruction[2])
        else:
            gate = one_qubit_gate({"H": H, "S": S, "Z": Z}[name], instruction[1])
        unitary = gate @ unitary
    return unitary


def pauli(word):
    table = {"I": I, "X": X, "Y": Y, "Z": Z}
    return kron3(*(table[c] for c in word))


def measure(state, word, outcome):
    projector = (np.eye(8) + outcome * pauli(word)) / 2
    branch = projector @ state
    probability = float(np.vdot(branch, branch).real)
    if probability > 1e-14:
        branch /= np.sqrt(probability)
    return probability, branch


# Tree nodes are (Pauli word, plus-child, minus-child); leaves are integer guesses.
TREE = (
    "XXZ",
    (
        "IXI",
        0,
        ("ZIX", 0, 3),
    ),
    (
        "ZZI",
        ("YXX", 1, 0),
        ("XXI", 2, 0),
    ),
)


def run_tree(state, node, accumulated=1.0):
    if isinstance(node, int):
        return [(node, accumulated)]
    word, plus_child, minus_child = node
    leaves = []
    for outcome, child in [(+1, plus_child), (-1, minus_child)]:
        probability, post = measure(state, word, outcome)
        if probability > 1e-14:
            leaves.extend(run_tree(post, child, accumulated * probability))
    return leaves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    unitary = toffoli() @ preprocessing_unitary()
    total_success = 0.0

    print("Perfect-discrimination verification")
    print("Clifford pre-processing gates:", len(CLIFFORD_SEQUENCE))
    print("Non-Clifford gates: one CCX(1,2 -> 3)")
    print("This verifies the CCX witness; T-state injection is not simulated.")
    print()

    for true_label, state in enumerate(STATES):
        transformed = unitary @ state
        leaves = run_tree(transformed, TREE)
        correct = sum(prob for guess, prob in leaves if guess == true_label)
        total = sum(prob for _, prob in leaves)
        if not np.isclose(total, 1.0, atol=1e-10, rtol=0.0):
            raise AssertionError(f"Branches do not sum to one for {LABELS[true_label]}")
        if not np.isclose(correct, 1.0, atol=1e-10, rtol=0.0):
            raise AssertionError(f"Protocol fails for {LABELS[true_label]}")
        total_success += correct / 4
        print(f"{LABELS[true_label]:6s}: total={total:.12f}, correct={correct:.12f}, leaves={leaves}")

    print()
    print(f"average success probability = {total_success:.12f}")
    if abs(total_success - 1.0) > 1e-10:
        raise AssertionError("Protocol is not perfect")
    print("verification passed")


if __name__ == "__main__":
    main()
