"""Independent, bounded validation of the generalized E8 XOR C++ backend.

Run with STABDISC_TEST_NATIVE=1. Compilation uses temporary directories; no
large XOR search is performed. The production C++ source is included in a test
harness so packed keys and conditional Bellman values can be checked directly.
"""

from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from stabdisc.pauli import enc, extend_code
from stabdisc.solver import success_probability
from stabdisc.states import e8_sign, xor_repetition


SOURCE = Path(__file__).resolve().parents[1] / "native/src/e8_xor_t2_raw_parallel.cpp"

HARNESS = r'''
#define main stabdisc_e8_program_main
#include "@SOURCE@"
#undef main

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]).rfind("--test-", 0) != 0)
        return stabdisc_e8_program_main(argc, argv);
    build_hypotheses();
    const std::string mode = argv[1];
    if (mode == "--test-hypotheses") {
        std::cout << N << ' ' << COEFF_DEN << ' ' << VALUE_DEN << '\n';
        for (int u = 0; u < UMAX; ++u) {
            const Val a = hypothesis_coefficient(0, u);
            const Val b = hypothesis_coefficient(1, u);
            if (a.a || a.b || b.a || b.b)
                std::cout << u << ' ' << a.a << ' ' << a.b << ' '
                          << b.a << ' ' << b.b << '\n';
        }
        return 0;
    }
    std::vector<int> rows;
    for (int i = 2; i < argc; ++i) rows.push_back(std::stoi(argv[i]));
    const Key raw = pack_rows(rows);
    if (basis_from_key(raw) != rows) throw std::logic_error("raw packing lost a row");
    const auto code = code_from_key(raw);
    const Key canonical = canonical_key(code);
    if (code_from_key(canonical) != code)
        throw std::logic_error("canonical packing lost a stabilizer");
    if (mode == "--test-key") {
        auto reverse_code = code;
        std::reverse(reverse_code.begin(), reverse_code.end());
        if (!(canonical_key(reverse_code) == canonical))
            throw std::logic_error("canonical key depends on code order");
        if (rows.size() > 1) {
            auto alternate = rows;
            alternate[0] = mulp(N, rows[0], rows[1]);
            if (!(canonical_key(code_from_key(pack_rows(alternate))) == canonical))
                throw std::logic_error("canonical key depends on generating set");
        }
        FlatMemo memo(20);
        memo.begin_epoch();
        memo.put(raw, {100, 0});
        for (size_t index = 0; index < rows.size(); ++index) {
            auto variant = rows;
            variant[index] ^= (1 << NB);
            const Key changed = pack_rows(variant);
            if (changed == raw) throw std::logic_error("signed row key collision");
            if (basis_from_key(changed) != variant)
                throw std::logic_error("high signed row failed round trip");
            memo.put(changed, {static_cast<int32_t>(index + 1), 0});
        }
        Val found;
        if (!memo.get(raw, found) || found.a != 100 || found.b != 0)
            throw std::logic_error("memo overwrote original wide key");
        for (size_t index = 0; index < rows.size(); ++index) {
            auto variant = rows;
            variant[index] ^= (1 << NB);
            if (!memo.get(pack_rows(variant), found) ||
                found.a != static_cast<int32_t>(index + 1) || found.b != 0)
                throw std::logic_error("memo conflated signed wide keys");
        }
        for (int row : basis_from_key(canonical)) std::cout << row << ' ';
        std::cout << '\n';
        for (int p : code) std::cout << p << ' ';
        std::cout << '\n';
        return 0;
    }
    if (mode == "--test-branch") {
        BranchSolver solver(20, 0, 0, 0);
        const Val value = solver.solve_code_for_test(code);
        std::cout << value.a << ' ' << value.b << ' ' << VALUE_DEN << '\n';
        return 0;
    }
    throw std::invalid_argument("unknown test mode");
}
'''


def subgroup(n, rows):
    code = (0,)
    for row in rows:
        code = extend_code(code, row & ((1 << (2 * n)) - 1), row >> (2 * n), n)
    return code


@unittest.skipUnless(os.environ.get("STABDISC_TEST_NATIVE") == "1" and shutil.which("c++"),
                     "set STABDISC_TEST_NATIVE=1 with a C++20 compiler for native checks")
class E8XorNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stabdisc-e8-validation-")
        cls.directory = Path(cls.temporary.name)
        cls.addClassCleanup(cls.temporary.cleanup)
        harness = cls.directory / "harness.cpp"
        harness.write_text(HARNESS.replace("@SOURCE@", SOURCE.as_posix()))
        cls.binaries = {}
        cls.ensembles = {}
        for copies in range(1, 5):
            binary = cls.directory / f"e8_xor_k{copies}"
            subprocess.run(["c++", "-std=c++20", "-O2", "-pthread", f"-DXOR_COPIES={copies}",
                            str(harness), "-o", str(binary)], check=True, capture_output=True,
                           text=True, timeout=120)
            cls.binaries[copies] = binary
            cls.ensembles[copies] = xor_repetition(e8_sign(), copies)

    def run_native(self, copies, *args, check=True, timeout=30):
        return subprocess.run([str(self.binaries[copies]), *map(str, args)], check=check,
                              capture_output=True, text=True, timeout=timeout,
                              cwd=self.directory)

    def test_all_hypothesis_coefficients_match_python_for_one_to_four_copies(self):
        for copies in range(1, 5):
            with self.subTest(copies=copies):
                lines = self.run_native(copies, "--test-hypotheses").stdout.splitlines()
                n, coefficient_denominator, value_denominator = map(int, lines[0].split())
                self.assertEqual(n, 3 * copies)
                self.assertEqual(coefficient_denominator, 4 ** copies)
                self.assertEqual(value_denominator, 2 ** (5 * copies + 1))
                actual = {}
                for line in lines[1:]:
                    u, a0, b0, a1, b1 = map(int, line.split())
                    self.assertNotIn(u, actual)
                    actual[u] = (a0, b0, a1, b1)
                ensemble = self.ensembles[copies]
                expected = {
                    u: tuple(int(part * coefficient_denominator)
                             for state in ensemble.expectations
                             for part in (state[u].a, state[u].b))
                    for u in ensemble.expectations[0]
                }
                self.assertEqual(len(actual), 16 ** copies + 1)
                self.assertEqual(actual, expected)

    def test_wide_signed_keys_preserve_every_row_and_distinguish_outcomes(self):
        for copies in range(1, 5):
            n = 3 * copies
            # Independent single-qubit X/Y/Z axes give an explicit canonical
            # basis. Its final signed rows cross bit 128 for k=3 and bit 256
            # for k=4, exercising the truncation in the former two-word key.
            rows = [enc(n, (1 << q) if q % 3 != 2 else 0,
                        (1 << q) if q % 3 != 0 else 0, q % 2)
                    for q in range(n)]
            with self.subTest(copies=copies, packed_bits=n * (2 * n + 1)):
                lines = self.run_native(copies, "--test-key", *rows).stdout.splitlines()
                actual_basis = tuple(map(int, lines[0].split()))
                actual_code = tuple(map(int, lines[1].split()))
                unsigned_mask = (1 << (2 * n)) - 1
                expected_basis = tuple(sorted(rows, key=lambda p: p & unsigned_mask, reverse=True))
                self.assertEqual(actual_basis, expected_basis)
                self.assertEqual(actual_code, subgroup(n, rows))

    def test_one_copy_full_root_search_matches_python(self):
        output = self.directory / "complete-k1.csv"
        result = self.run_native(1, "--output", output, "--no-resume", "--capacity-power", 20,
                                 "--progress-every", 0)
        summaries = [line for line in result.stdout.splitlines() if line.startswith("RESULT_NUM ")]
        self.assertEqual(len(summaries), 1)
        _, a, b, _, denominator = summaries[0].split()
        self.assertEqual(int(b), 0)
        actual = Fraction(int(a), int(denominator))
        self.assertEqual(actual, Fraction(3, 4))
        self.assertEqual(success_probability(self.ensembles[1]), actual)
        rows = [line for line in output.read_text().splitlines() if line and line[0].isdigit()]
        self.assertEqual({int(line.split(",")[0]) for line in rows}, set(range(1, 64)))

    def test_two_copy_conditional_values_match_legacy_and_python(self):
        from stabdisc.pauli import canonical_key
        from stabdisc.solver import StabilizerDiscriminationSolver
        solver = StabilizerDiscriminationSolver(self.ensembles[2])
        # Legacy numerators were independently recomputed using the original
        # two-copy solver, reversing qubits within each E8 register to match
        # this package's first-character-is-qubit-0 convention.
        cases = [((1, 2, 520, 1024), 80),
                 ((9, 576, 18, 1152), 88),
                 ((4105, 576, 18, 5248), 72)]
        for rows, legacy_numerator in cases:
            with self.subTest(rows=rows):
                a, b, denominator = map(int, self.run_native(2, "--test-branch", *rows).stdout.split())
                self.assertEqual((a, b, denominator), (legacy_numerator, 0, 2048))
                key = canonical_key(subgroup(6, rows), 6)
                self.assertEqual(solver.value(key), Fraction(a, denominator))

    def test_high_rank_nine_and_twelve_qubit_values_match_direct_measurement(self):
        for copies in (3, 4):
            n = 3 * copies
            coefficient_denominator = 4 ** copies
            weights = [{u: int(value.a * coefficient_denominator) for u, value in state.items()}
                       for state in self.ensembles[copies].expectations]
            rows = [enc(n, 1 << q, 0, q % 2) for q in range(n - 1)]
            code = subgroup(n, rows)
            mask = (1 << (2 * n)) - 1

            def stop_value(group):
                totals = [sum((-1 if p >> (2 * n) else 1) * state.get(p & mask, 0)
                              for p in group) for state in weights]
                return Fraction(max(totals), 2 * len(group) * coefficient_denominator)

            candidates = [stop_value(code)]
            for x, z in ((1, 0), (0, 1), (1, 1)):
                q = enc(n, x << (n - 1), z << (n - 1))
                candidates.append(sum(stop_value(extend_code(code, q, sign, n)) for sign in (0, 1)))
            with self.subTest(copies=copies):
                a, b, denominator = map(int, self.run_native(copies, "--test-branch", *rows).stdout.split())
                self.assertEqual(b, 0)
                self.assertEqual(Fraction(a, denominator), max(candidates))

    def test_bounded_three_and_four_copy_runs_do_not_report_an_optimum(self):
        for copies in (3, 4):
            with self.subTest(copies=copies):
                output = self.directory / f"capped-k{copies}.csv"
                result = self.run_native(copies, "--q-start", 1, "--q-end", 1, "--output", output,
                                         "--no-resume", "--capacity-power", 20, "--node-limit", 32,
                                         "--progress-every", 8, check=False)
                self.assertGreater(result.returncode, 0, "a capped search must exit cleanly with an error")
                self.assertIn("limit", result.stderr.lower())
                self.assertIn("branch_completed=", result.stderr)
                self.assertNotIn("RESULT_NUM", result.stdout)
                self.assertNotIn("Exact raw root search complete", result.stdout)

    def test_completed_single_root_remains_a_partial_result(self):
        output = self.directory / "single-k1.csv"
        result = self.run_native(1, "--q-start", 1, "--q-end", 1, "--output", output,
                                 "--no-resume", "--capacity-power", 20, "--progress-every", 0)
        self.assertNotIn("RESULT_NUM", result.stdout)
        self.assertIn("lower bound", result.stdout.lower())

    def test_resume_and_combine_reject_incompatible_or_corrupt_records(self):
        output = self.directory / "audit-k1.csv"
        self.run_native(1, "--q-start", 1, "--q-end", 1, "--output", output,
                        "--no-resume", "--capacity-power", 20, "--progress-every", 0)
        original = output.read_text()
        row = next(line for line in original.splitlines() if line and line[0].isdigit())
        parts = row.split(",")
        conflicting = parts.copy()
        conflicting[1] = "48" if int(parts[1]) == 32 else "32"
        conflicting[3] = conflicting[5] = str(int(conflicting[1]) // 2)
        conflicting = ",".join(conflicting)
        wrong_sum = parts.copy()
        wrong_sum[3] = str(int(wrong_sum[3]) + 1)
        invalid_root = parts.copy()
        invalid_root[0] = "64"
        variants = {
            "copies": original.replace("# COPIES=1", "# COPIES=2"),
            "denominator": original.replace("# VALUE_DEN=64", "# VALUE_DEN=2048"),
            "label_order": original.replace("# LABEL_ORDER=python_v1", "# LABEL_ORDER=legacy"),
            "branch_sum": original.replace(row, ",".join(wrong_sum)),
            "root_label": original.replace(row, ",".join(invalid_root)),
            "duplicate": original + conflicting + "\n",
        }
        for name, data in variants.items():
            bad = self.directory / f"invalid-{name}.csv"
            bad.write_text(data)
            for arguments in (("--combine", bad),
                              ("--output", bad, "--q-start", 1, "--q-end", 1,
                               "--capacity-power", 20)):
                with self.subTest(problem=name, operation=arguments[0]):
                    result = self.run_native(1, *arguments, check=False)
                    self.assertGreater(result.returncode, 0)
                    self.assertNotIn("RESULT_NUM", result.stdout)
        conflicting_file = self.directory / "conflicting-k1.csv"
        conflicting_file.write_text(original.replace(row, conflicting))
        result = self.run_native(1, "--combine", output, conflicting_file, check=False)
        self.assertGreater(result.returncode, 0)
        self.assertIn("Conflicting duplicate", result.stderr)
        self.assertNotIn("RESULT_NUM", result.stdout)

    def test_empty_output_receives_metadata_and_can_resume(self):
        output = self.directory / "empty-k1.csv"
        output.touch()
        arguments = ("--q-start", 1, "--q-end", 1, "--output", output,
                     "--capacity-power", 20, "--progress-every", 0)
        self.run_native(1, *arguments)
        saved = output.read_text()
        self.assertIn("# COPIES=1", saved)
        self.assertIn("# VALUE_DEN=64", saved)
        self.assertIn("# LABEL_ORDER=python_v1", saved)
        result = self.run_native(1, *arguments)
        self.assertEqual(saved, output.read_text())
        self.assertIn("already completed = 1", result.stdout)
        self.assertNotIn("RESULT_NUM", result.stdout)


if __name__ == "__main__":
    unittest.main()
