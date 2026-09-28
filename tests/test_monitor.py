import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stabdisc.monitor import RunProgressReader, latest_run, main


HEADER = "# VALUE_DEN=16\nq,a,b,plus_a,plus_b,minus_a,minus_b,nodes_plus,nodes_minus,seconds\n"


def row(q, a=8):
    return f"{q},{a},0,4,0,{a-4},0,3,4,0.1\n"


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = self.directory / "run.json"
        self.manifest.write_text(json.dumps({"status": "running", "required_root_count": 15}))
        self.csv = self.directory / "roots.csv"
        self.csv.write_text(HEADER)

    def test_incremental_rows_wait_for_newline_and_ignore_duplicates(self):
        self.csv.write_text(HEADER + row(1) + row(2)[:-3])
        reader = RunProgressReader(self.directory)
        event = reader.poll()
        self.assertEqual((event.completed, event.total, event.best_probability), (1, 15, 0.5))
        with self.csv.open("a") as output:
            output.write(row(2)[-3:] + row(1) + row(3, 12))
        event = reader.poll()
        self.assertEqual((event.completed, event.best_probability), (3, 0.75))
        self.assertEqual(reader.poll().completed, 3)

    def test_full_coverage_does_not_claim_verified_completion(self):
        self.csv.write_text(HEADER + "".join(row(q) for q in range(1, 16)))
        reader = RunProgressReader(self.directory)
        event = reader.poll()
        self.assertEqual(event.completed, 15)
        self.assertEqual(event.phase, "solving")
        self.assertIn("awaiting final verification", event.message)
        self.assertIn("process activity not checked", event.message)
        self.manifest.write_text(json.dumps({"status": "computed_native", "probability_a": "1/2", "probability_b": "0", "elapsed_seconds": 3}))
        self.assertEqual(reader.poll().phase, "complete")
        self.assertEqual(reader.elapsed_seconds, 3)

    def test_partial_metadata_keeps_last_snapshot(self):
        reader = RunProgressReader(self.directory)
        self.assertEqual(reader.poll().total, 15)
        self.manifest.write_text('{"status":')
        event = reader.poll()
        self.assertEqual(event.total, 15)
        self.assertEqual(reader.status, "running")

    def test_worker_counters_are_never_added_to_global_count(self):
        stderr = self.directory / "stderr.log"
        stderr.write_text("worker=0 branch_completed=3500000 memo=3000000\n")
        reader = RunProgressReader(self.directory)
        event = reader.poll()
        self.assertEqual(event.completed, 0)
        self.assertIsNone(event.solved_projectors)
        self.assertIn("last reported branch: worker 0, 3,500,000 states solved", event.message)
        with stderr.open("a") as output:
            output.write("worker=1 branch_completed=500000 memo=400000\n")
        self.assertIn("worker 1, 500,000", reader.poll().message)

    def test_standalone_counts_have_no_fictitious_total(self):
        self.manifest.write_text(json.dumps({"status": "running", "required_root_count": None}))
        self.csv.unlink()
        (self.directory / "stderr.log").write_text("completed=10000 rank=3 memo=5000 elapsed=2s\n")
        event = RunProgressReader(self.directory, n_qubits=5, standalone=True).poll()
        self.assertIsNone(event.total)
        self.assertIsNone(event.completed)
        self.assertEqual(event.solved_projectors, 10000)

    def test_failed_run_keeps_last_branch_diagnostic(self):
        self.manifest.write_text(json.dumps({"status": "failed", "error": ""}))
        (self.directory / "stderr.log").write_text("worker=0 branch_completed=3500000 memo=3000000\n")
        event = RunProgressReader(self.directory).poll()
        self.assertEqual(event.phase, "failed")
        self.assertIn("last reported branch", event.message)
        self.assertIn("3,500,000", event.message)

    def test_runner_can_omit_stale_status_note(self):
        reader = RunProgressReader(self.directory)
        self.assertNotIn("activity not checked", reader.poll(include_status_note=False).message)

    def test_invalid_rows_and_denominator_are_not_counted(self):
        self.csv.write_text(HEADER + row(1).replace(",8,0,", ",20,0,") + row(16) + row(2))
        event = RunProgressReader(self.directory).poll()
        self.assertEqual(event.completed, 1)
        self.assertIn("invalid", event.message)
        self.csv.write_text(HEADER.replace("=16", "=32") + row(1))
        event = RunProgressReader(self.directory, denominator=16).poll()
        self.assertEqual(event.completed, 0)
        self.assertIsNone(event.best_probability)
        self.assertIn("denominator mismatch", event.message)

    def test_file_truncation_resets_cached_counts(self):
        self.csv.write_text(HEADER + row(1) + row(2))
        reader = RunProgressReader(self.directory)
        self.assertEqual(reader.poll().completed, 2)
        self.csv.write_text(HEADER)
        self.assertEqual(reader.poll().completed, 0)

    def test_once_is_read_only(self):
        self.csv.write_text(HEADER + row(1))
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = main(["--run-dir", str(self.directory), "--once"])
        self.assertEqual(result, 0)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.directory.iterdir()}
        self.assertEqual(before, after)
        self.assertIn("1/15 roots", output.getvalue())
        self.assertNotIn("Complete", output.getvalue())

    def test_ctrl_c_only_closes_monitor(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output), patch("stabdisc.monitor.time.sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(main(["--run-dir", str(self.directory)]), 0)
        self.assertIn("computation was not stopped", output.getvalue())
        self.assertEqual(json.loads(self.manifest.read_text())["status"], "running")

    def test_latest_uses_existing_native_run_directories(self):
        child = self.directory / "example"
        child.mkdir()
        (child / "run.json").write_text('{}')
        self.assertEqual(latest_run(self.directory), child.resolve())
        with self.assertRaises(FileNotFoundError):
            latest_run(self.directory / "missing")


if __name__ == "__main__":
    unittest.main()
