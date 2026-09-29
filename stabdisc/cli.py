"""Fresh success-probability calculations from source-defined ensembles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import __version__
from .solver import StabilizerDiscriminationSolver
from .progress import TerminalProgress
from .states import (e6_full, e6_sign, e8_full, e8_sign, four_state,
                     tensor_ensembles, twoqubit_mixture, xor_repetition)

EXAMPLES = (
    "e8_full", "e8_sign", "e8_xor", "e6_full", "e6_sign", "e6_xor",
    "twoqubit_tstates", "twoqubit_xor", "four_state_tstates",
)


def _nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _positive(text):
    value = _nonnegative(text)
    if value == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _arguments(parser):
    parser.add_argument("--tstates", type=_nonnegative, default=0,
                        help="number of common T resources (default: 0)")
    parser.add_argument("--copies", type=_positive,
                        help="independent copies (XOR default: 2; otherwise: 1)")
    parser.add_argument("--backend", choices=("auto", "python", "native"), default="auto",
                        help="auto: Python through 3 qubits, native for larger supported examples")
    parser.add_argument("--output", type=Path, help="write result JSON to a new file")
    parser.add_argument("--policy", type=Path, help="write a Python measurement tree to a new file")
    parser.add_argument("--max-projectors", type=_positive,
                        help="Python memo-entry budget; exceeding it aborts without an optimum")
    parser.add_argument("--allow-large", action="store_true",
                        help="explicitly allow a Python search above four qubits")
    parser.add_argument("--threads", type=_positive, default=1,
                        help="native worker count (default: 1)")
    parser.add_argument("--run-dir", type=Path,
                        help="new native run folder (default: unique folder inside runs/)")
    parser.add_argument("--build-dir", type=Path,
                        help="native compilation folder (default: build/ beside this project)")
    parser.add_argument("--capacity-power", type=_positive,
                        help="native memo table has 2**P slots per worker")
    parser.add_argument("--timeout", type=float,
                        help="native execution time limit in seconds, excluding compilation")
    parser.add_argument("--node-limit", type=_positive,
                        help="E8 XOR native solved-state limit per outcome branch; exceeding it aborts without an optimum")
    parser.add_argument("--no-progress", action="store_true",
                        help="hide the live terminal progress display")


def _spec(name, tstates, copies):
    """Create a recipe without allocating a large tensor-product ensemble."""
    if name not in EXAMPLES:
        raise ValueError(f"unknown example: {name}")
    xor = name.endswith("_xor")
    copies = (2 if xor else 1) if copies is None else copies
    if xor:
        if tstates:
            raise ValueError("XOR scripts do not take --tstates; combine resources with the Python API")
        factory, width = {
            "e8_xor": (e8_sign, 3), "e6_xor": (e6_sign, 3),
            "twoqubit_xor": (twoqubit_mixture, 2),
        }[name]
        return lambda: xor_repetition(factory(), copies), width * copies, copies
    if name == "e8_full" and copies > 1:
        if tstates:
            raise ValueError("combine independent full-label copies and T resources with the Python API")
        return lambda: tensor_ensembles(*(e8_full() for _ in range(copies))), 3 * copies, copies
    if copies != 1:
        raise ValueError("--copies is supported for XOR and E8 full-label examples")
    factory, width = {
        "e8_full": (e8_full, 3), "e8_sign": (e8_sign, 3),
        "e6_full": (e6_full, 3), "e6_sign": (e6_sign, 3),
        "twoqubit_tstates": (twoqubit_mixture, 2),
        "four_state_tstates": (four_state, 3),
    }[name]
    return lambda: factory(tstates=tstates), width + tstates, copies


def _write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _evaluate(name, args):
    from .native import native_available, run

    factory, qubits, copies = _spec(name, args.tstates, args.copies)
    for path in (args.output, args.policy):
        if path is not None and path.exists():
            raise ValueError(f"choose a new output path; {path} already exists")
    if args.output and args.policy and args.output.resolve() == args.policy.resolve():
        raise ValueError("--output and --policy must name different files")
    backend = args.backend
    if backend == "auto":
        backend = ("native" if qubits > 3 and args.policy is None
                   and native_available(name, args.tstates, copies) else "python")
    if backend == "python":
        if qubits > 4 and not args.allow_large:
            raise ValueError(
                f"this needs an exhaustive {qubits}-qubit Python search, with no applicable native "
                "backend selected; use --allow-large only if you intend this expensive calculation"
            )
        if any(option is not None for option in
               (args.run_dir, args.build_dir, args.capacity_power, args.timeout, args.node_limit)) or args.threads != 1:
            raise ValueError("native run/compile options require --backend native")
    else:
        if not native_available(name, args.tstates, copies):
            raise ValueError("no native implementation for these parameters; use --backend python")
        if args.policy is not None or args.max_projectors is not None or args.allow_large:
            raise ValueError("--policy, --max-projectors and --allow-large are Python options")
    record = {
        "example": name, "backend": backend, "qubits": qubits,
        "copies": copies, "tstates": args.tstates, "version": __version__,
        "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"Starting a fresh {qubits}-qubit calculation with the {backend} backend.", file=sys.stderr)
    if backend == "native" and qubits >= 5:
        print("This is an exhaustive large search and can require substantial time and memory.",
              file=sys.stderr)
    started = time.perf_counter()
    with TerminalProgress(enabled=not args.no_progress) as display:
        callback = None if args.no_progress else display.update
        if backend == "python":
            solver = StabilizerDiscriminationSolver(
                factory(), record_policy=args.policy is not None, max_projectors=args.max_projectors,
                progress=callback,
            )
            result = solver.solve()
            value = result.success_probability
            record.update(status="computed_python", arithmetic=result.arithmetic,
                          solved_projectors=result.solved_projectors)
        else:
            result = run(name, tstates=args.tstates, copies=copies, threads=args.threads,
                         run_dir=args.run_dir, build_dir=args.build_dir,
                         capacity_power=args.capacity_power, timeout=args.timeout,
                         node_limit=args.node_limit,
                         progress=callback)
            value = result.probability
            record.update(status="computed_native", arithmetic="exact",
                          root_count=result.root_count, run_dir=str(result.run_dir),
                          search="all first Pauli measurements" if result.root_count is not None
                          else "standalone exhaustive recursion")
    if args.policy is not None:
        _write_json(args.policy, {"example": name, "policy": solver.policy_tree(),
                                  "qubit_order": "first character is qubit 0"})
        record["policy_file"] = str(args.policy)
    record.update(probability=float(value), exact=str(value), a=str(value.a), b=str(value.b),
                  field="Q(sqrt(2))", elapsed_seconds=time.perf_counter() - started)
    if args.output is not None:
        _write_json(args.output, record)
    print(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))
    return record


def run_example(name, argv=None):
    parser = argparse.ArgumentParser(description=f"Compute {name} from scratch.")
    _arguments(parser)
    args = parser.parse_args(argv)
    try:
        return _evaluate(name, args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Calculation interrupted.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute a stabilizer-discrimination success probability from scratch.")
    parser.add_argument("example", choices=EXAMPLES)
    _arguments(parser)
    args = parser.parse_args(argv)
    try:
        _evaluate(args.example, args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Calculation interrupted.\n")


if __name__ == "__main__":
    main()
