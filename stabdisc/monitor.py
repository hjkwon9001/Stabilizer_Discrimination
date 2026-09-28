"""Read-only progress monitoring for both new and already-running native jobs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from .algebra import Qsqrt2
from .progress import ProgressUpdate, TerminalProgress

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HEADER = ["q", "a", "b", "plus_a", "plus_b", "minus_a", "minus_b", "nodes_plus", "nodes_minus", "seconds"]


class _Tail:
    """Read appended complete lines only; survive truncation and replacement."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.identity = None
        self.pending = b""

    def read(self) -> tuple[list[str], bool]:
        try:
            with self.path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = self.identity is not None and (identity != self.identity or stat.st_size < self.offset)
                if reset:
                    self.offset, self.pending = 0, b""
                self.identity = identity
                handle.seek(self.offset)
                data = handle.read()
                self.offset = handle.tell()
        except OSError:
            return [], False
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        return [line.decode("utf-8", errors="replace").rstrip("\r") for line in lines], reset


class RunProgressReader:
    """Incremental snapshot reader that never modifies or controls a run.

    CSV rows are partial evidence, not an independent completion certificate.
    Only the parent runner's verified ``computed_native`` metadata produces a
    complete event. An old ``running`` status does not establish process liveness.
    """

    def __init__(self, run_dir: str | Path, *, denominator: int | None = None,
                 total: int | None = None, n_qubits: int | None = None,
                 standalone: bool = False):
        self.run_dir = Path(run_dir).resolve()
        self.denominator = denominator
        self.total = total if total is not None else (4 ** n_qubits - 1 if n_qubits is not None and not standalone else None)
        self.metadata: dict = {}
        self.status = "unknown"
        self._csv = _Tail(self.run_dir / "roots.csv")
        self._stderr = _Tail(self.run_dir / "stderr.log")
        self._roots: set[int] = set()
        self._best: Qsqrt2 | None = None
        self._header = False
        self._denominator_matches = True
        self._warning = ""
        self._diagnostic = ""
        self._states: int | None = None
        try:
            self._started = (self.run_dir / "run.json").stat().st_mtime
        except OSError:
            self._started = time.time()

    @property
    def elapsed_seconds(self) -> float:
        value = self.metadata.get("elapsed_seconds")
        if self.status in {"computed_native", "failed", "interrupted"} and isinstance(value, (int, float)) and math.isfinite(value):
            return max(0.0, value)
        return max(0.0, time.time() - self._started)

    def _read_metadata(self):
        try:
            value = json.loads((self.run_dir / "run.json").read_text())
            if isinstance(value, dict):
                self.metadata = value
        except (OSError, ValueError):
            # A writer may be between truncation and its next full JSON write.
            pass
        self.status = str(self.metadata.get("status", "unknown"))
        total = self.metadata.get("required_root_count")
        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
            self.total = total
        denominator = self.metadata.get("denominator")
        if self.denominator is None and isinstance(denominator, int) and not isinstance(denominator, bool) and denominator > 0:
            self.denominator = denominator
        started = self.metadata.get("started_utc", self.metadata.get("started_at"))
        if isinstance(started, str):
            try:
                parsed = datetime.fromisoformat(started.replace("Z", "+00:00"))
                self._started = parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
            except ValueError:
                pass

    def _read_roots(self):
        lines, reset = self._csv.read()
        if reset:
            self._roots.clear()
            self._best, self._header, self._warning = None, False, ""
            self._denominator_matches = True
        for line in lines:
            if line.startswith("#"):
                match = re.search(r"\bVALUE_DEN\s*=\s*(\d+)", line, re.I)
                if match:
                    denominator = int(match[1])
                    if denominator <= 0 or self.denominator not in (None, denominator):
                        self._warning = "CSV denominator mismatch; progress values unavailable"
                        self._header = False
                        self._denominator_matches = False
                    else:
                        self.denominator = denominator
                continue
            if not line.strip():
                continue
            fields = next(csv.reader([line]))
            if not self._header:
                self._header = fields == _HEADER
                continue
            try:
                if not self._denominator_matches:
                    continue
                if len(fields) != len(_HEADER) or not self.denominator:
                    raise ValueError("unexpected columns")
                q, a, b, pa, pb, ma, mb, np, nm = map(int, fields[:-1])
                seconds = float(fields[-1])
                if q < 1 or (self.total is not None and q > self.total) or min(np, nm) < 0 or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid root diagnostics")
                if (pa + ma, pb + mb) != (a, b):
                    raise ValueError("invalid root sum")
                probability = Qsqrt2(Fraction(a, self.denominator), Fraction(b, self.denominator))
                if not Qsqrt2(0) <= probability <= Qsqrt2(1):
                    raise ValueError("invalid probability")
                for ba, bb in ((pa, pb), (ma, mb)):
                    if not Qsqrt2(0) <= Qsqrt2(Fraction(ba, self.denominator), Fraction(bb, self.denominator)) <= Qsqrt2(1):
                        raise ValueError("invalid branch probability")
            except (ValueError, ZeroDivisionError):
                self._warning = "Ignoring an invalid complete CSV row; final validation still required"
                continue
            if q in self._roots:
                continue
            self._roots.add(q)
            if self._best is None or probability > self._best:
                self._best = probability

    def _read_diagnostics(self):
        lines, reset = self._stderr.read()
        if reset:
            self._diagnostic, self._states = "", None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "branch_completed=" in line:
                # Each worker/branch counter resets; it is not a global total.
                match = re.search(r"worker=(\d+).*?branch_completed=(\d+)", line)
                self._diagnostic = (f"last reported branch: worker {match[1]}, {int(match[2]):,} states solved"
                                    if match else line[:160])
            elif re.match(r"completed=\d+\s+rank=", line):
                self._states = int(re.match(r"completed=(\d+)", line)[1])
            else:
                self._diagnostic = line[:160]

    def poll(self, *, include_status_note: bool = True) -> ProgressUpdate:
        self._read_metadata()
        self._read_roots()
        self._read_diagnostics()
        phase = {"computed_native": "complete", "failed": "failed", "interrupted": "interrupted"}.get(self.status, "solving")
        best = self._best
        if phase == "complete":
            try:
                best = Qsqrt2(Fraction(self.metadata["probability_a"]), Fraction(self.metadata["probability_b"]))
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                pass
        messages = []
        if phase == "solving":
            if include_status_note:
                messages.append(f"recorded status: {self.status}; process activity not checked")
            if self.total is not None and len(self._roots) == self.total:
                messages.append("all roots recorded; awaiting final verification")
        if self._warning:
            messages.append(self._warning)
        if phase in {"failed", "interrupted"}:
            messages.append(str(self.metadata.get("error") or "See stderr.log for diagnostics"))
        if self._diagnostic and phase != "complete":
            messages.append(self._diagnostic)
        return ProgressUpdate(phase, backend="native", completed=len(self._roots) if self.total is not None or self._header else None,
                              total=self.total, solved_projectors=self._states,
                              best_probability=float(best) if best is not None else None,
                              message="; ".join(messages), run_dir=str(self.run_dir))


def latest_run(runs_dir: str | Path) -> Path:
    """Return the most recently modified run manifest, without editing it."""
    directory = Path(runs_dir).expanduser().resolve()
    candidates = []
    if directory.is_dir():
        for path in directory.iterdir():
            try:
                manifest = path / "run.json"
                if path.is_dir() and manifest.is_file():
                    candidates.append((manifest.stat().st_mtime_ns, str(path), path))
            except OSError:
                continue
    if not candidates:
        raise FileNotFoundError(f"No runs with run.json found in {directory}")
    return max(candidates)[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observe an existing native run without restarting or modifying it.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--latest", action="store_true", help="monitor the latest run under runs/")
    selection.add_argument("--run-dir", type=Path, help="monitor this specific existing run directory")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs", help="directory searched by --latest")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots (default: 1)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    try:
        directory = latest_run(args.runs_dir) if args.latest else args.run_dir.expanduser().resolve()
        if not directory.is_dir() or not (directory / "run.json").is_file():
            raise FileNotFoundError(f"Not a native run directory with run.json: {directory}")
    except OSError as exc:
        parser.error(str(exc))
    reader = RunProgressReader(directory)
    event = reader.poll()
    print(f"Monitoring {directory}\nRead-only monitor; Ctrl+C closes only this display.", file=sys.stderr)
    progress = TerminalProgress(interval=0.2, elapsed_seconds=reader.elapsed_seconds)
    progress.update(event)
    if args.once or event.phase in {"complete", "failed", "interrupted"}:
        progress.close()
        return 0
    progress.start()
    try:
        while event.phase not in {"complete", "failed", "interrupted"}:
            time.sleep(args.interval)
            event = reader.poll()
            progress.update(event)
    except KeyboardInterrupt:
        progress.close()
        print("Monitor closed. The computation was not stopped.", file=sys.stderr)
        return 0
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
