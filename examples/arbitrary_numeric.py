"""Compute success for user-supplied density matrices (requires NumPy).

Edit ``states`` and ``priors`` below to use your own ensemble. This example
constructs |0><0| and |+><+| and optimizes their adaptive Pauli discrimination.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stabdisc import density_matrices_to_ensemble, success_probability


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    zero = np.array([1.0, 0.0], dtype=complex)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    states = [np.outer(zero, zero.conj()), np.outer(plus, plus.conj())]
    priors = [0.5, 0.5]
    ensemble = density_matrices_to_ensemble(states, priors=priors, labels=["zero", "plus"])
    probability = success_probability(ensemble)
    print(f"Stabilizer success probability: {float(probability):.12f}")


if __name__ == "__main__":
    main()
