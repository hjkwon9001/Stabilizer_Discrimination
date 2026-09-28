import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from stabdisc import native
from stabdisc.algebra import Qsqrt2
from stabdisc.solver import success_probability
from stabdisc.states import e6_sign, four_state, twoqubit_mixture


def root_table(n=2, denominator=16):
    """Synthetic audit table: diagnostic checks must not need saved results."""
    rows = [f"# VALUE_DEN={denominator}",
            "q,a,b,plus_a,plus_b,minus_a,minus_b,nodes_plus,nodes_minus,seconds"]
    rows.extend(f"{q},8,0,4,0,4,0,1,1,0" for q in range(1, 4 ** n))
    return "\n".join(rows) + "\n"


class NativeValidationTests(unittest.TestCase):
    def test_supported_native_combinations(self):
        cases = [("e8_sign", m, 1) for m in (2, 3)]
        cases += [("e6_sign", m, 1) for m in range(4)]
        cases += [("twoqubit_tstates", m, 1) for m in range(5)]
        cases += [("four_state_tstates", m, 1) for m in range(4)]
        cases += [("twoqubit_xor", 0, k) for k in (2, 3)]
        cases += [("e6_full", 3, 1), ("e6_xor", 0, 2), ("e8_xor", 0, 2)]
        self.assertEqual(len(cases), 20)
        for name, tstates, copies in cases:
            with self.subTest(name=name, tstates=tstates, copies=copies):
                self.assertTrue(native.native_available(name, tstates, copies))
                configuration = native._configuration(name, tstates, copies)
                self.assertTrue((native.PROJECT_ROOT / "native" / "src" / configuration.source).is_file())
        for name, tstates, copies in (("e8_full", 0, 1), ("e8_sign", 1, 1),
                                     ("e6_full", 0, 1), ("twoqubit_xor", 0, 1),
                                     ("e6_sign", 4, 1), ("e8_xor", 1, 2),
                                     ("twoqubit_tstates", 0, 2), ("unknown", 0, 1)):
            self.assertFalse(native.native_available(name, tstates, copies))

    def test_strict_summary_parser(self):
        self.assertEqual(native._parse_result("RESULT_NUM 8 0 DEN 16\n", 16), Qsqrt2(1) / 2)
        for output in ("partial only", "RESULT_NUM 14 0 DEN 32", "RESULT_NUM 17 0 DEN 16",
                       "RESULT_NUM 8 0 DEN 16\nRESULT_NUM 8 0 DEN 16"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                native._parse_result(output, 16)

    def test_full_root_coverage(self):
        table = root_table()
        values = native._parse_root_table(table, 16, 2)
        self.assertEqual(len(values), 15)
        self.assertEqual(max(values), Qsqrt2(1) / 2)
        for bad in ("\n".join(table.splitlines()[:-1]), table + table.splitlines()[-1] + "\n",
                    table.replace("15,8,", "16,8,")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                native._parse_root_table(bad, 16, 2)

    def test_corrupt_root_arithmetic_and_diagnostics(self):
        table = root_table()
        bad_tables = [table.replace("VALUE_DEN=16", "VALUE_DEN=32"),
                      table.replace("1,8,0,4,0,4,0,1,1,0", "1,8,0,5,0,4,0,1,1,0"),
                      table.replace("1,8,0,4,0,4,0,1,1,0", "1,8,0,-1,0,9,0,1,1,0"),
                      table.replace("1,8,0,4,0,4,0,1,1,0", "1,8,0,4,0,4,0,-1,1,0"),
                      table.replace("1,8,0,4,0,4,0,1,1,0", "1,8,0,4,0,4,0,1,1,nan")]
        for bad in bad_tables:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                native._parse_root_table(bad, 16, 2)

    def test_existing_directory_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(native, "_checked_build") as build:
                with self.assertRaises(FileExistsError):
                    native.run("twoqubit_tstates", run_dir=directory)
                build.assert_not_called()

    def test_invalid_parameters_rejected_before_build(self):
        for args in ({"threads": 0}, {"threads": True}, {"capacity_power": 7},
                     {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
                     {"tstates": True}, {"copies": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                native.run("twoqubit_tstates", **args)

    def test_partial_table_rejected_despite_success_sentinel(self):
        table = "\n".join(root_table().splitlines()[:-1]) + "\n"

        def fake_run(command, **kwargs):
            self.assertIn("--no-resume", command)
            self.assertNotIn("--q-file", command)
            self.assertNotIn("--state-file", command)
            self.assertEqual(command[command.index("--q-end") + 1], "15")
            (Path(kwargs["cwd"]) / "roots.csv").write_text(table)
            kwargs["stdout"].write("RESULT_NUM 8 0 DEN 16\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            with patch.object(native, "_checked_build", return_value=(Path("/fake/solver"), {})), \
                 patch.object(native.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "Incomplete root set"):
                    native.run("twoqubit_tstates", run_dir=directory)
            self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "failed")

    def test_exact_summary_disagreement_rejected(self):
        def fake_run(command, **kwargs):
            (Path(kwargs["cwd"]) / "roots.csv").write_text(root_table())
            kwargs["stdout"].write("RESULT_NUM 9 0 DEN 16\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(native, "_checked_build", return_value=(Path("/fake/solver"), {})), \
                 patch.object(native.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "summary differs"):
                    native.run("twoqubit_tstates", run_dir=Path(temporary) / "run")

    @staticmethod
    def _script(directory, body):
        executable = directory / "fake_solver"
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o755)
        return executable

    def test_streamed_progress_precedes_exact_validation(self):
        updates = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = root_table().splitlines(keepends=True)
            executable = self._script(directory,
                "import time\n"
                "with open('roots.csv', 'w') as output:\n"
                f"    output.write({''.join(table[:7])!r})\n"
                "    output.flush()\n"
                "    time.sleep(0.3)\n"
                f"    output.write({''.join(table[7:])!r})\n"
                "print('RESULT_NUM 8 0 DEN 16')\n")
            with patch.object(native, "_checked_build", return_value=(executable, {})):
                result = native.run("twoqubit_tstates", run_dir=directory / "run", progress=updates.append)
            self.assertEqual(result.probability, Qsqrt2(1) / 2)
            partial = [item for item in updates if item.phase == "solving" and item.completed == 5]
            self.assertTrue(partial)
            self.assertEqual(partial[-1].total, 15)
            self.assertEqual(partial[-1].best_probability, 0.5)
            self.assertEqual(updates[-2].phase, "validating")
            self.assertEqual(updates[-1].phase, "complete")
            self.assertEqual(updates[-1].completed, 15)
            metadata = json.loads((result.run_dir / "run.json").read_text())
            self.assertIn("started_utc", metadata)
            self.assertIn("pid", metadata)

    def test_streamed_corrupt_result_never_reports_completion(self):
        updates = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = self._script(directory,
                f"open('roots.csv', 'w').write({root_table()!r})\n"
                "print('RESULT_NUM 9 0 DEN 16')\n")
            with patch.object(native, "_checked_build", return_value=(executable, {})):
                with self.assertRaisesRegex(ValueError, "summary differs"):
                    native.run("twoqubit_tstates", run_dir=directory / "run", progress=updates.append)
            self.assertNotIn("complete", [item.phase for item in updates])
            self.assertEqual(updates[-1].phase, "failed")

    def test_timeout_and_interrupt_reap_native_child(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                executable = self._script(directory, "import time\ntime.sleep(30)\n")
                updates = []

                def progress(update):
                    updates.append(update)
                    if interrupt and update.phase == "solving":
                        raise KeyboardInterrupt

                expected = KeyboardInterrupt if interrupt else subprocess.TimeoutExpired
                with patch.object(native, "_checked_build", return_value=(executable, {})):
                    with self.assertRaises(expected):
                        native.run("twoqubit_tstates", run_dir=directory / "run", timeout=0.2,
                                   progress=progress)
                metadata = json.loads((directory / "run" / "run.json").read_text())
                self.assertEqual(metadata["status"], "interrupted" if interrupt else "failed")
                self.assertTrue(metadata["error"])
                self.assertEqual(updates[-1].phase, metadata["status"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(metadata["pid"], 0)

    @unittest.skipUnless(os.environ.get("STABDISC_TEST_NATIVE") == "1" and shutil.which("c++"),
                         "Set STABDISC_TEST_NATIVE=1 to compile and run three small C++ cases")
    def test_native_matches_independent_python_fresh_computation(self):
        cases = [("twoqubit_tstates", twoqubit_mixture, 15),
                 ("e6_sign", e6_sign, 63), ("four_state_tstates", four_state, 63)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, factory, root_count in cases:
                with self.subTest(name=name):
                    result = native.run(name, run_dir=directory / name, build_dir=directory / "build", timeout=60)
                    self.assertEqual(result.probability, success_probability(factory()))
                    self.assertEqual(result.root_count, root_count)
                    metadata = json.loads((result.run_dir / "run.json").read_text())
                    self.assertEqual(metadata["status"], "computed_native")
                    self.assertNotIn("--q-file", metadata["command"])


if __name__ == "__main__":
    unittest.main()
