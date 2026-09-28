"""Compile the bundled C++ solvers and compute from scratch.

Every raw solver evaluates all nonidentity first Pauli measurements. No
reference answers, root lists, state files, or previous run outputs are read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Callable
from uuid import uuid4

from .algebra import Qsqrt2
from .monitor import RunProgressReader
from .progress import ProgressUpdate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NativeResult:
    probability: Qsqrt2
    root_count: int | None
    run_dir: Path


@dataclass(frozen=True)
class _Configuration:
    case_id: str
    source: str
    defines: tuple[tuple[str, int], ...]
    n_qubits: int
    denominator: int
    standalone: bool = False


def _configuration(name: str, tstates: int, copies: int) -> _Configuration:
    if isinstance(tstates, bool) or not isinstance(tstates, int) or tstates < 0:
        raise ValueError("tstates must be a nonnegative integer")
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("copies must be a positive integer")
    if copies == 1:
        m = tstates
        if name == "twoqubit_tstates" and m <= 4:
            n = 2 + m
            return _Configuration(f"twoq_t_m{m}", "twoqubit_tstate_discrimination.cpp",
                                  (("TSTATES", m),), n, 2 * 2 ** (1 + (m + 1) // 2) * 2 ** n)
        if name == "e6_sign" and m <= 3:
            n = 3 + m
            return _Configuration(f"e6_t_m{m}", "e6_tstate_discrimination.cpp",
                                  (("TSTATES", m),), n, 6 * 2 ** ((m + 1) // 2) * 2 ** n)
        if name == "four_state_tstates" and m <= 3:
            n = 3 + m
            return _Configuration(f"four_state_m{m}", "four_state_tstate_discrimination.cpp",
                                  (("TSTATES", m),), n, 4 * 2 ** m * 2 ** n)
        if name == "e6_full" and m == 3:
            return _Configuration("e6_full_t3", "e6_full_t3_raw_parallel.cpp", (), 6, 3072)
        if name == "e8_sign" and m in (2, 3):
            return _Configuration(f"e8_sign_m{m}",
                                  "e8_sign_t2_exact.cpp" if m == 2 else "e8_sign_t3_raw_parallel.cpp",
                                  (), 3 + m, 512 if m == 2 else 2048, standalone=m == 2)
    if tstates == 0:
        if name == "twoqubit_xor" and copies in (2, 3):
            n = 2 * copies
            # The built-in pair has expectation denominator 2. A=(rho0+rho1)/2
            # and D=(rho0-rho1)/2 have denominator 4 before tensor products.
            return _Configuration(f"twoq_xor_k{copies}", "twoqubit_xor_raw_parallel.cpp",
                                  (("XOR_COPIES", copies),), n, 2 * 4 ** copies * 2 ** n)
        if name in ("e6_xor", "e8_xor") and copies == 2:
            source = "e6_sign_xor_t2_raw_parallel.cpp" if name == "e6_xor" else "e8_xor_t2_raw_parallel.cpp"
            return _Configuration(f"{name}_k2", source, (), 6, 1152 if name == "e6_xor" else 2048)
    raise ValueError(f"No native solver for {name!r}, tstates={tstates}, copies={copies}; use the Python solver")


def native_available(name: str, tstates: int = 0, copies: int = 1) -> bool:
    """Whether this example and parameter combination has a C++ implementation.

    This checks support, not whether a C++20 compiler is installed.
    """
    try:
        _configuration(name, tstates, copies)
    except ValueError:
        return False
    return True


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stop_process(process: subprocess.Popen) -> None:
    """Reap our process and, on POSIX, its compiler/solver process group."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _run_with_progress(command: list[str], *, stdout, stderr, cwd: Path | None = None,
                       timeout: float | None = None, tick: Callable[[], None],
                       on_start: Callable[[int], None] | None = None) -> int:
    """Poll file-backed diagnostics without risking a blocked output pipe."""
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=cwd,
                               start_new_session=os.name == "posix")
    try:
        if on_start is not None:
            on_start(process.pid)
        while True:
            tick()
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                if process.poll() is None:
                    raise subprocess.TimeoutExpired(command, timeout)
            try:
                result = process.wait(timeout=0.1 if remaining is None else max(0.001, min(0.1, remaining)))
                tick()  # Include even very fast jobs' final flushed root rows.
                return result
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _stop_process(process)
        raise


def build(name: str, *, tstates: int = 0, copies: int = 1,
          build_dir: str | Path | None = None,
          progress: Callable[[ProgressUpdate], None] | None = None) -> Path:
    """Build a supported solver, or reuse a build matching source and options.

    Requires a C++20 compiler named by CXX (one executable) or ``c++``.
    Compilation never reads or computes any reference success probabilities.
    """
    configuration = _configuration(name, tstates, copies)
    return _checked_build(configuration, build_dir, progress=progress)[0]


def _checked_build(configuration: _Configuration, build_dir: str | Path | None,
                   progress: Callable[[ProgressUpdate], None] | None = None) -> tuple[Path, dict]:
    source = PROJECT_ROOT / "native" / "src" / configuration.source
    if not source.is_file():
        raise FileNotFoundError(f"Missing C++ source: {source}; keep native/ alongside stabdisc/")
    directory = Path(build_dir).resolve() if build_dir is not None else PROJECT_ROOT / "build"
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / configuration.case_id
    manifest = directory / f"{configuration.case_id}.build.json"
    source_hash = _hash(source)
    defines = dict(configuration.defines)
    selected = os.environ.get("CXX") or "c++"
    compiler = shutil.which(selected)
    if compiler is None:
        raise FileNotFoundError(f"C++20 compiler not found: {selected}; on macOS install Command Line Tools with xcode-select --install")
    if executable.is_file() and manifest.is_file():
        cached = False
        try:
            metadata = json.loads(manifest.read_text())
            cached = (metadata.get("case_id") == configuration.case_id
                      and metadata.get("source_sha256") == source_hash
                      and metadata.get("defines") == defines
                      and metadata.get("compiler") == compiler
                      and metadata.get("executable_sha256") == _hash(executable))
        except (ValueError, OSError):
            pass
        if cached:
            if progress is not None:
                progress(ProgressUpdate(phase="ready", backend="native", message="Using an existing matching solver build"))
            return executable, metadata
    temporary = directory / f".{configuration.case_id}.{uuid4().hex}.building"
    command = [compiler, "-O3", "-std=c++20", "-pthread"]
    command += [f"-D{key}={value}" for key, value in configuration.defines]
    command += [str(source), "-o", str(temporary)]
    log = directory / f"{configuration.case_id}.compiler.log"
    try:
        if progress is None:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            log.write_text(completed.stdout + completed.stderr)
            returncode = completed.returncode
        else:
            update = ProgressUpdate(phase="compiling", backend="native", message="Building the C++ solver")
            with log.open("w") as output:
                returncode = _run_with_progress(command, stdout=output, stderr=output,
                                                tick=lambda: progress(update))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Native compilation failed ({returncode}); see {log}")
    temporary.replace(executable)
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
    metadata = {"case_id": configuration.case_id, "source": str(source),
                "source_sha256": source_hash, "defines": defines, "compiler": compiler,
                "compiler_version": version.stdout, "command": command,
                "executable_sha256": _hash(executable)}
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    if progress is not None:
        progress(ProgressUpdate(phase="ready", backend="native", message="C++ solver ready"))
    return executable, metadata


def _parse_result(stdout: str, denominator: int) -> Qsqrt2:
    matches = re.findall(r"^RESULT_NUM\s+(-?\d+)\s+(-?\d+)\s+DEN\s+(\d+)\s*$", stdout, re.M)
    if len(matches) != 1:
        raise ValueError("Expected exactly one complete RESULT_NUM line")
    a, b, den = map(int, matches[0])
    if den != denominator:
        raise ValueError(f"Native denominator mismatch: {den} != {denominator}")
    probability = Qsqrt2(Fraction(a, den), Fraction(b, den))
    if not Qsqrt2(0) <= probability <= Qsqrt2(1):
        raise ValueError("Native probability is outside [0,1]")
    return probability


def _parse_root_table(text: str, denominator: int, n_qubits: int) -> tuple[Qsqrt2, ...]:
    """Validate exact arithmetic and all 4**n-1 distinct first Pauli labels."""
    declarations = [int(value) for value in re.findall(r"\bvalue_den\s*=\s*(\d+)", text, re.I)]
    if not declarations or any(value != denominator for value in declarations):
        raise ValueError("CSV denominator is absent or inconsistent with the solver")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    header = ["q", "a", "b", "plus_a", "plus_b", "minus_a", "minus_b", "nodes_plus", "nodes_minus", "seconds"]
    if reader.fieldnames != header:
        raise ValueError("Unexpected root-result CSV header")
    values, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed root-result CSV row")
        try:
            root, a, b, pa, pb, ma, mb, np, nm = (int(row[key]) for key in header[:-1])
            seconds = float(row["seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed numeric root-result CSV field") from exc
        if not 1 <= root < 4 ** n_qubits or root in seen:
            raise ValueError(f"Invalid or duplicate root ID: {root}")
        if (pa + ma, pb + mb) != (a, b):
            raise ValueError(f"Branch sums do not match for root {root}")
        if min(np, nm) < 0 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid diagnostics for root {root}")
        for branch_a, branch_b in ((pa, pb), (ma, mb)):
            branch = Qsqrt2(Fraction(branch_a, denominator), Fraction(branch_b, denominator))
            if not Qsqrt2(0) <= branch <= Qsqrt2(1):
                raise ValueError(f"Root {root} has a branch value outside [0,1]")
        probability = Qsqrt2(Fraction(a, denominator), Fraction(b, denominator))
        if not Qsqrt2(0) <= probability <= Qsqrt2(1):
            raise ValueError(f"Root {root} has a value outside [0,1]")
        values.append(probability)
        seen.add(root)
    wanted = set(range(1, 4 ** n_qubits))
    if seen != wanted:
        raise ValueError(f"Incomplete root set: found {len(seen)} of {len(wanted)} first Pauli measurements")
    return tuple(values)


def run(name: str, *, tstates: int = 0, copies: int = 1, threads: int = 1,
        run_dir: str | Path | None = None, build_dir: str | Path | None = None,
        capacity_power: int | None = None, timeout: float | None = None,
        progress: Callable[[ProgressUpdate], None] | None = None) -> NativeResult:
    """Compute a success probability from scratch in a new run directory.

    Raw solvers check all 4**n-1 first Pauli measurements and validate the full
    resulting table before returning. E8 sign with two T states uses a standalone
    exhaustive Bellman solver; it has no per-root CSV and returns root_count=None.
    No previous results are loaded. A new directory is generated under runs/ when
    run_dir is omitted. An explicitly supplied existing directory is rejected.

    timeout bounds computation, not compilation. Five- and six-qubit searches
    can be very expensive; threads also multiply each worker's memory use.
    An optional progress callback receives build status and completed first
    measurements read from the solver's freshly written logs. Partial best
    probabilities are lower bounds; the final result is always validated.
    """
    configuration = _configuration(name, tstates, copies)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if capacity_power is not None and (isinstance(capacity_power, bool) or not isinstance(capacity_power, int) or not 8 <= capacity_power <= 31):
        raise ValueError("capacity_power must be an integer in [8,31]")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if configuration.standalone and (capacity_power is not None or threads != 1):
        raise ValueError("E8 sign with two T states does not accept thread or capacity settings")
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = PROJECT_ROOT / "runs" / f"{configuration.case_id}-{stamp}-{uuid4().hex[:8]}"
    else:
        directory = Path(run_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"Run directory already exists; choose a new directory to keep this computation fresh: {directory}")
    executable, build_metadata = _checked_build(configuration, build_dir, progress=progress)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(executable)]
    if not configuration.standalone:
        power = capacity_power if capacity_power is not None else {2: 8, 3: 10, 4: 13, 5: 19, 6: 25}[configuration.n_qubits]
        command += ["--threads", str(threads), "--capacity-power", str(power),
                    "--q-start", "1", "--q-end", str(4 ** configuration.n_qubits - 1),
                    "--output", str(directory / "roots.csv"), "--no-resume"]
    metadata = {"name": name, "tstates": tstates, "copies": copies,
                "status": "running", "command": command, "build": build_metadata,
                "n_qubits": configuration.n_qubits, "denominator": configuration.denominator,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "root_coverage": "standalone_bellman" if configuration.standalone else "all_nonidentity_paulis",
                "required_root_count": None if configuration.standalone else 4 ** configuration.n_qubits - 1}
    manifest = directory / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            if progress is None:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=directory, timeout=timeout, check=False)
                returncode = completed.returncode
            else:
                reader = RunProgressReader(directory, denominator=configuration.denominator,
                                           n_qubits=configuration.n_qubits, standalone=configuration.standalone)

                def record_pid(pid: int) -> None:
                    metadata["pid"] = pid
                    manifest.write_text(json.dumps(metadata, indent=2) + "\n")

                returncode = _run_with_progress(command, stdout=stdout, stderr=stderr, cwd=directory,
                                                timeout=timeout, tick=lambda: progress(reader.poll(include_status_note=False)),
                                                on_start=record_pid)
        if returncode:
            raise RuntimeError(f"Native solver exited with status {returncode}; see {directory / 'stderr.log'}")
        if progress is not None:
            progress(ProgressUpdate(phase="validating", backend="native", run_dir=str(directory),
                                    message="Checking exact arithmetic and measurement coverage"))
        output = (directory / "stdout.log").read_text()
        probability = _parse_result(output, configuration.denominator)
        count = None
        if configuration.standalone:
            if not re.search(r"^byrank r0=1(?:\s|$)", output, re.M):
                raise ValueError("Standalone solver did not report completion of the initial Bellman state")
        else:
            values = _parse_root_table((directory / "roots.csv").read_text(), configuration.denominator, configuration.n_qubits)
            if max(values) != probability:
                raise ValueError("Native summary differs from the exact root-table maximum")
            count = len(values)
        metadata.update(status="computed_native", probability_a=str(probability.a), probability_b=str(probability.b),
                        root_count=count, elapsed_seconds=time.monotonic() - started)
        metadata["output_sha256"] = {item: _hash(directory / item) for item in ("stdout.log", "stderr.log", "roots.csv") if (directory / item).exists()}
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        if progress is not None:
            progress(ProgressUpdate(phase="complete", backend="native", completed=count, total=count,
                                    best_probability=float(probability), run_dir=str(directory), message="Exact result verified"))
        return NativeResult(probability, count, directory)
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = "Interrupted by user" if isinstance(exc, KeyboardInterrupt) else str(exc)
        metadata.update(status=status, error=error, elapsed_seconds=time.monotonic() - started)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        if progress is not None:
            try:
                progress(ProgressUpdate(phase=status, backend="native", run_dir=str(directory), message=error))
            except BaseException:
                pass  # Preserve the computation error if a callback itself failed.
        raise
