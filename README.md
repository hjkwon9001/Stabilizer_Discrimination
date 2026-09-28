# Compute stabilizer-discrimination probabilities from scratch

This is a standalone, source-only project. It contains the Python library,
example scripts, C++ source for larger calculations, and tests. It contains
**no saved probability tables, root-orbit lists, compiled programs, or prior
run outputs**. Every example computes its result.

## Start here

Open Terminal and enter:

```bash
cd ~/Desktop/Stabilizer_Success_Source
python3 run.py e8_sign
```

This performs a fresh three-qubit computation and returns an exact probability
of `3/4` and decimal `0.75`. You can also run the example directly:

```bash
python3 examples/e8_sign.py
```

Keep the project folder intact. Scripts locate the accompanying library
automatically, including when launched by full path from another directory.
Installing `stabdisc` is not needed to run these scripts.

## Live progress

Progress is displayed automatically while a calculation runs. Use the same
commands as before:

```bash
python3 run.py twoqubit_xor --copies 2
python3 run.py e8_xor --copies 2
```

The native display shows a progress bar, completed first measurements out of
the total, elapsed time, and the best success probability found so far.
For a long first measurement, branch-state updates show that the search is
advancing even when the completed-measurement count is still zero.
Compilation and final result verification are separate phases.

The percentage counts **completed first measurements**, not elapsed work or
time remaining: different measurements can take very different amounts of
time. The best-so-far value is an achievable lower bound until the search
finishes and its result is verified.

The Python solver and the standalone E8 two-T solver show visited search
states and elapsed time. Their eventual state count depends on pruning, so
they do not display a guessed completion percentage.

The display updates in place in Terminal. Redirected output uses occasional
plain status lines, and the final JSON result remains on standard output.
To suppress the progress display:

```bash
python3 run.py twoqubit_xor --copies 2 --no-progress
```

To watch a native calculation from a second Terminal window, including a
calculation started before this update:

```bash
cd ~/Desktop/Stabilizer_Success_Source
python3 monitor.py --latest
```

To select a run explicitly or read one snapshot:

```bash
python3 monitor.py --run-dir runs/YOUR_RUN_FOLDER
python3 monitor.py --latest --once
```

The monitor only reads the run's files. Pressing Ctrl-C in the monitor stops
the display and leaves the calculation alone. Pressing Ctrl-C in the
calculation's own terminal cancels that calculation. An interrupted or failed
run is labeled accordingly; a full-looking bar alone is not reported as a
verified result.

Python 3.10 or newer and NumPy are required. If NumPy is missing:

```bash
python3 -m pip install -r requirements.txt
```

To use a separate Python environment instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py e8_sign
```

## XOR examples

Two-copy XOR discrimination of the two-qubit mixtures:

```bash
python3 run.py twoqubit_xor --copies 2
```

The program compiles its C++ source on first use and performs a fresh search
over all 255 first Pauli measurements. Its result is `13/16 = 0.8125`.

The identical calculation can be launched through its example script:

```bash
python3 examples/twoqubit_xor.py --copies 2
```

For E6 and E8:

```bash
python3 examples/e6_xor.py --copies 2
python3 examples/e8_xor.py --copies 2
```

These are **large six-qubit searches** over all 4095 possible first
measurements. They can require gigabytes of memory and very long runtimes,
and are substantially slower than calculations using symmetry reductions.
This project recomputes the full search; it does not read a saved answer or
saved list of symmetry representatives. To check the setup quickly, use
`--copies 1`.

## All examples

| Script | Calculation | Parameters |
|---|---|---|
| `examples/e8_full.py` | E8 full labels | `--tstates M`; or independent labels with `--copies K` |
| `examples/e8_sign.py` | E8 binary sign | `--tstates M` |
| `examples/e8_xor.py` | E8 sign parity | `--copies K` |
| `examples/e6_full.py` | E6 full labels | `--tstates M` |
| `examples/e6_sign.py` | E6 binary sign | `--tstates M` |
| `examples/e6_xor.py` | E6 sign parity | `--copies K` |
| `examples/twoqubit_tstates.py` | Two-qubit mixture pair | `--tstates M` |
| `examples/twoqubit_xor.py` | Two-qubit mixture parity | `--copies K` |
| `examples/four_state_tstates.py` | Four full labels | `--tstates M` |
| `examples/arbitrary_numeric.py` | Arbitrary input density matrices | Edit the matrices in the script |
| `examples/verify_four_state_one_toffoli.py` | A specified Clifford+Toffoli circuit | A circuit witness, not an optimizer or T-injection simulation |

XOR defaults to two independent copies. Other examples default to zero T
states and one copy. Each XOR task asks for the parity of independent binary
labels, not repeated copies of the same unknown label.

Examples with a small common resource:

```bash
python3 examples/twoqubit_tstates.py --tstates 1
python3 examples/four_state_tstates.py
python3 examples/e6_sign.py
```

## Choosing the computation backend

The default `--backend auto` selects Python through three qubits. For larger
supported examples it compiles and uses the native C++ implementation.

```bash
python3 examples/twoqubit_xor.py --copies 2 --backend python
python3 examples/twoqubit_xor.py --copies 2 --backend native
```

The Python solver accepts arbitrary ensembles and performs the complete
Bellman recursion. The native implementations are specialized:

| Native calculation | Supported parameters |
|---|---|
| E8 sign | 2 or 3 T states |
| E8 XOR | 2 copies |
| E6 sign | 0–3 T states |
| E6 full labels | 3 T states |
| E6 XOR | 2 copies |
| Two-qubit T assistance | 0–4 T states |
| Two-qubit XOR | 2 or 3 copies |
| Four-state T assistance | 0–3 T states |

Unsupported native parameter choices can be computed with the generic Python
solver, subject to available time and memory. Python searches beyond four
qubits require `--allow-large`. This does not make them fast.

Native calculations need a C++20 compiler, such as Apple Clang or GCC.
On a Mac without a compiler, install Apple's command-line tools:

```bash
xcode-select --install
```

No prebuilt executable is supplied. A locally compiled executable may be
reused on subsequent runs, but **probabilities are always recomputed**.
Each native run uses a new output directory; previous CSVs are never resumed.
All first Pauli measurements must finish before the result is accepted.

Native files are generated in `build/` and `runs/` only when you run the
code. To choose locations and a runtime limit:

```bash
python3 examples/twoqubit_xor.py --copies 2 --threads 1 --timeout 60 --run-dir runs/my-xor-test
```

Use a new `--run-dir` each time. `--timeout` excludes compilation time.
The standalone E8 two-T solver uses its own fixed memory settings.

## Save a result or a measurement policy

```bash
python3 examples/e8_sign.py --output outputs/e8-sign.json
python3 examples/four_state_tstates.py --policy outputs/four-state-policy.json
```

Output paths must be new. The policy option uses the Python solver and
exports the selected Pauli measurements and guesses.

The JSON contains the decimal `probability` and exact coefficients `a,b`
meaning `a+b*sqrt(2)`. It records which backend actually computed the result.

## Reusable library

```python
from stabdisc import e8_sign, success_probability

probability = success_probability(e8_sign())
print(probability)         # 3/4, computed by the solver
print(float(probability))  # 0.75
```

To define your own exact ensemble:

```python
from fractions import Fraction
from stabdisc import Ensemble, success_probability

ensemble = Ensemble.from_pauli_dicts(
    n=1,
    states=[{"I": 1, "Z": 1}, {"I": 1, "X": 1}],
    priors=[Fraction(1, 2), Fraction(1, 2)],
    labels=["zero", "plus"],
)
print(success_probability(ensemble))
```

Use `Fraction` and `Qsqrt2` to retain exact arithmetic. Floating input
produces a numerical result. `density_matrices_to_ensemble` checks dense
input matrices; `append_t_states`, `tensor_ensembles`, and
`xor_repetition` construct resources and repeated tasks.

For Pauli words and dense tensors, the first character/factor is qubit 0.
The sparse integer representation uses that qubit as the least significant
mask bit. Physicality checks on dense matrices are numerical, even for exact
Pauli coefficients.

If importing from notebooks or programs outside this folder, install the
library in that Python environment with `python3 -m pip install -e .`.
With an installed wheel, native builds require access to this source project;
the direct source-folder workflow above is the intended use.

## Algorithm

At a signed stabilizer-code branch `S`, the solver compares guessing the
most likely label with all nontrivial commuting logical Pauli refinements:

```text
w_i(S) = p_i Tr(Pi_S rho_i)
V(S) = max(max_i w_i(S), max_q [V(S_(q,+)) + V(S_(q,-))])
```

At rank equal to the number of qubits, the solver guesses a label. Exact
branch weights and comparisons lie in `Q(sqrt(2))` for all built-in inputs.
The generic implementation memoizes signed projectors and can export a
measurement tree. See `stabdisc/solver.py`, `states.py`, and `pauli.py`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The ordinary tests use analytic expectations and independent dense-matrix
calculations, rather than reading stored result tables. Optional native
tests compile C++ and run small fresh calculations in temporary directories:

```bash
STABDISC_TEST_NATIVE=1 python3 -m unittest discover -s tests -v
```

## Folder layout

```text
Stabilizer_Success_Source/
  run.py             simple entry point
  monitor.py         read-only viewer for a native run
  stabdisc/          computation library
  examples/          individual runnable examples
  native/src/        C++ source only
  tests/             correctness and standalone-execution tests
  requirements.txt   Python dependency
  pyproject.toml     optional package installation
```
