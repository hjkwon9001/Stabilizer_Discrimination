"""Check examples in an independent copied source tree, without installation."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

try:
    import numpy as np
except ImportError:
    np = None


FAMILIES = (
    "e8_full", "e8_sign", "e8_xor", "e6_full", "e6_sign", "e6_xor",
    "twoqubit_tstates", "twoqubit_xor", "four_state_tstates",
)


@unittest.skipIf(np is None, "NumPy is required to run the examples")
class PortableExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stabdisc source tests ")
        cls.addClassCleanup(cls.temporary.cleanup)
        temporary = Path(cls.temporary.name)
        cls.source = temporary / "independent source with spaces"
        original = Path(__file__).resolve().parents[1]
        for name in ("stabdisc", "examples"):
            shutil.copytree(
                original / name,
                cls.source / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copyfile(original / "run.py", cls.source / "run.py")
        cls.unrelated_cwd = temporary / "unrelated working directory"
        cls.unrelated_cwd.mkdir()
        cls.installed = temporary / "other installed packages"
        (cls.installed / "stabdisc").mkdir(parents=True)
        (cls.installed / "stabdisc" / "__init__.py").write_text(
            'raise AssertionError("An installed package shadowed the adjacent source")\n',
            encoding="utf-8",
        )

    def run_script(self, name, *arguments, check=True):
        # Exclude user-site packages, editable installations and PYTHONPATH.
        # Only the NumPy dependency and a conflicting installed package are
        # exposed. Each example must explicitly locate its own sibling package.
        launcher = textwrap.dedent("""\
            from pathlib import Path
            import runpy
            import sys

            script, numpy_location, installed, *arguments = sys.argv[1:]
            sys.path.append(numpy_location)
            sys.path.insert(0, installed)
            if Path(script).name == "run.py":
                # Match normal ``python /path/to/run.py`` script-directory setup.
                sys.path.insert(0, str(Path(script).parent))
            sys.argv = [script, *arguments]
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
            if Path(script).stem != "verify_four_state_one_toffoli":
                import stabdisc
                root = Path(script).parent if Path(script).name == "run.py" else Path(script).parents[1]
                expected = root / "stabdisc" / "__init__.py"
                assert Path(stabdisc.__file__).resolve() == expected.resolve(), stabdisc.__file__
            """)
        script = self.source / "run.py" if name == "run" else self.source / "examples" / f"{name}.py"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", launcher,
             str(script),
             str(Path(np.__file__).resolve().parent.parent), str(self.installed),
             *arguments],
            cwd=self.unrelated_cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_all_examples_explain_usage_from_an_independent_copy(self):
        for name in (*FAMILIES, "arbitrary_numeric", "verify_four_state_one_toffoli"):
            with self.subTest(example=name):
                help_text = self.run_script(name, "--help").stdout
                self.assertIn("usage:", help_text)
                self.assertNotIn("--mode", help_text)
                self.assertNotIn("--verify-reference", help_text)
                if name in FAMILIES:
                    self.assertIn("--backend", help_text)

    def test_small_examples_recompute_exact_probabilities(self):
        for name, arguments, expected in (
            ("twoqubit_tstates", (), "7/8"),
            ("twoqubit_xor", ("--copies", "1"), "7/8"),
            ("e8_sign", (), "3/4"),
        ):
            with self.subTest(example=name):
                output = self.run_script(name, "--backend", "python", *arguments).stdout
                result = json.loads(output)
                self.assertEqual(result["a"], expected)
                self.assertEqual(result["b"], "0")
                self.assertEqual(result["status"], "computed_python")
                self.assertEqual(result["backend"], "python")
                self.assertGreater(result["solved_projectors"], 0)

    def test_numeric_example_computes_the_one_qubit_result(self):
        output = self.run_script("arbitrary_numeric").stdout
        self.assertIn("Stabilizer success probability: 0.750000000000", output)

    def test_root_runner_from_an_independent_copy(self):
        self.assertIn("usage:", self.run_script("run", "--help").stdout)
        output = self.run_script("run", "twoqubit_tstates", "--backend", "python").stdout
        result = json.loads(output)
        self.assertEqual(result["a"], "7/8")
        self.assertEqual(result["status"], "computed_python")

    def test_witness_recomputes_branch_probabilities(self):
        output = self.run_script("verify_four_state_one_toffoli").stdout
        self.assertIn("average success probability = 1.000000000000", output)
        self.assertIn("verification passed", output)

    def test_saved_result_mode_is_unavailable(self):
        result = self.run_script("twoqubit_tstates", "--mode", "saved", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
