"""Lightweight terminal progress without changing a solver's numerical work."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import shutil
import sys
import threading
import textwrap
import time
from typing import TextIO


@dataclass(frozen=True)
class ProgressUpdate:
    """A solver snapshot; totals are omitted when the search size is unknown.

    ``completed`` counts finished first measurements, while
    ``solved_projectors`` counts Bellman states when the solver provides that
    diagnostic. ``best_probability`` is only a lower bound until verification.
    """

    phase: str
    backend: str = ""
    completed: int | None = None
    total: int | None = None
    solved_projectors: int | None = None
    best_probability: float | None = None
    message: str = ""
    run_dir: str | None = None


_TERMINAL_PHASES = {"complete", "completed", "computed_native", "failed", "interrupted"}


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _plain(text: str) -> str:
    """Keep file diagnostics from introducing terminal control characters."""
    return " ".join("".join(c if c.isprintable() else " " for c in str(text)).split())


class TerminalProgress:
    """Display elapsed time and truthful solver progress on stderr.

    Use as a callback or call ``update(snapshot)``. A context manager runs a
    heartbeat, so a long native branch still has an advancing elapsed clock.
    Redirected output is plain text, rate limited to once per ten seconds apart
    from phase changes. Closing the display never stops a solver process.
    """

    def __init__(self, enabled: bool = True, stream: TextIO | None = None,
                 interval: float = 0.2, *, elapsed_seconds: float = 0.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("progress interval must be finite and positive")
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        try:
            self._tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self._tty = False
        self._started = time.monotonic() - max(0.0, elapsed_seconds)
        self._latest = ProgressUpdate("starting")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_render = -math.inf
        self._last_phase: str | None = None
        self._last_width = 0
        self._last_rows = 0
        self._last_line: str | None = None
        self._closed = False

    def start(self):
        if self.enabled and self._thread is None and not self._closed:
            self._thread = threading.Thread(target=self._heartbeat, name="stabdisc-progress", daemon=True)
            self._thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.close("interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed")
        else:
            self.close(None if self._latest.phase.lower() in _TERMINAL_PHASES else "complete")
        return False

    def __call__(self, update: ProgressUpdate):
        self.update(update)

    def update(self, update: ProgressUpdate):
        if not isinstance(update, ProgressUpdate):
            raise TypeError("progress update must be a ProgressUpdate")
        with self._lock:
            if self._closed:
                return
            self._latest = update
            self._render()

    def close(self, final_phase: str | None = None):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.3, self.interval * 2))
        with self._lock:
            if self._closed:
                return
            if final_phase is not None:
                self._latest = replace(self._latest, phase=final_phase)
            self._render(force=True)
            if self.enabled and self._tty:
                self.stream.write("\n")
                self.stream.flush()
            self._closed = True

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                self._render()

    def _format(self, now: float) -> str:
        event = self._latest
        terminal = event.phase.lower() in _TERMINAL_PHASES
        spinner = "" if terminal else "|/-\\"[int((now - self._started) / self.interval) % 4] + " "
        label = _plain(event.phase.replace("_", " ")).capitalize()
        if event.backend:
            label += f" ({_plain(event.backend)})"
        parts = [spinner + label]
        if event.completed is not None and event.total is not None and event.total > 0:
            fraction = max(0.0, min(1.0, event.completed / event.total))
            filled = int(fraction * 18)
            parts.append(f"[{'#' * filled}{'-' * (18 - filled)}] {event.completed:,}/{event.total:,} roots ({fraction:.1%})")
        elif event.completed is not None:
            parts.append(f"{event.completed:,} roots")
        if event.solved_projectors is not None:
            parts.append(f"{event.solved_projectors:,} states solved")
        parts.append(f"elapsed {_duration(now - self._started)}")
        if event.best_probability is not None:
            verified = event.phase.lower() in {"complete", "completed", "computed_native"}
            parts.append(f"{'success probability' if verified else 'best so far (lower bound)'} {float(event.best_probability):.10g}")
        if event.message:
            parts.append(_plain(event.message))
        return " | ".join(parts)

    def _render(self, force: bool = False):
        if not self.enabled:
            return
        now = time.monotonic()
        changed_phase = self._latest.phase != self._last_phase
        period = self.interval if self._tty else max(10.0, self.interval)
        if not force and not changed_phase and now - self._last_render < period:
            return
        line = self._format(now)
        if self._tty:
            # Keep each physical row inside the terminal width; a lone CR
            # cannot erase a previous update that wrapped onto several rows.
            width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            rows: list[str] = []
            for part in line.split(" | "):
                if rows and len(rows[-1]) + len(part) + 3 <= width:
                    rows[-1] += " | " + part
                else:
                    rows.extend(textwrap.wrap(part, width=width, break_long_words=True, break_on_hyphens=False) or [""])
            for index in range(self._last_rows):
                self.stream.write("\r\x1b[2K")
                if index < self._last_rows - 1:
                    self.stream.write("\x1b[1A")
            self.stream.write("\r" + "\n".join(rows))
            self._last_rows = len(rows)
        else:
            # Spinner animation has no value in a redirected log.
            if line[:2] in {"| ", "/ ", "- ", "\\ "}:
                line = line[2:]
            if line == self._last_line:
                return
            self.stream.write(line + "\n")
        self.stream.flush()
        self._last_line = line
        self._last_render, self._last_phase = now, self._latest.phase
