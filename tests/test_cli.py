"""Fresh-only interface behavior and computation-backend selection."""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from stabdisc import Qsqrt2
from stabdisc.cli import run_example


class CliTests(unittest.TestCase):
    def test_auto_uses_python_for_small_examples(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run_example("e8_sign", [])
        self.assertEqual(result["backend"], "python")
        self.assertEqual(result["a"], "3/4")
        self.assertNotIn("reference_status", result)
        self.assertNotIn("matches_reference", result)

    def test_auto_dispatches_two_copy_xor_to_native(self):
        native_result = SimpleNamespace(
            probability=Qsqrt2(Fraction(13, 16)), root_count=255, run_dir=Path("new-run"),
        )
        with patch("stabdisc.native.run", return_value=native_result) as run:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = run_example("twoqubit_xor", [])
        self.assertEqual(result["backend"], "native")
        self.assertEqual(result["root_count"], 255)
        self.assertEqual(run.call_args.args, ("twoqubit_xor",))
        self.assertEqual(run.call_args.kwargs["copies"], 2)

    def test_policy_selects_python_and_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = run_example("twoqubit_tstates", ["--policy", str(policy)])
            self.assertEqual(result["backend"], "python")
            self.assertTrue(policy.is_file())

    def test_native_failure_does_not_return_a_saved_answer(self):
        with patch("stabdisc.native.run", side_effect=RuntimeError("compiler unavailable")):
            with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_example("twoqubit_xor", [])
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(output.getvalue(), "")

    def test_progress_keeps_stdout_as_json(self):
        import json
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as status:
            result = run_example("twoqubit_tstates", [])
        self.assertEqual(json.loads(output.getvalue())["probability"], result["probability"])
        self.assertIn("elapsed", status.getvalue().lower())

    def test_no_progress_does_not_pass_a_callback(self):
        native_result = SimpleNamespace(
            probability=Qsqrt2(Fraction(13, 16)), root_count=255, run_dir=Path("new-run"),
        )
        with patch("stabdisc.native.run", return_value=native_result) as run:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_example("twoqubit_xor", ["--no-progress"])
        self.assertIsNone(run.call_args.kwargs["progress"])

    def test_output_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_example("e8_sign", ["--output", str(output)])
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
