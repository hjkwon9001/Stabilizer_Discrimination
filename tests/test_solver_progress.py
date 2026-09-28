"""Progress reports do not change exact optimization or invent totals."""

from fractions import Fraction
import unittest
from unittest.mock import patch

from stabdisc import (SolverLimitError, StabilizerDiscriminationSolver,
                      e8_sign, twoqubit_mixture)


class SolverProgressTests(unittest.TestCase):
    def test_progress_retains_exact_result_and_monotone_lower_bound(self):
        updates = []
        clock = iter(range(100_000))
        with patch("stabdisc.solver.time.monotonic", side_effect=lambda: next(clock)):
            solver = StabilizerDiscriminationSolver(twoqubit_mixture(1), progress=updates.append)
            result = solver.solve()
        self.assertEqual(result.success_probability.a, Fraction(3, 4))
        self.assertEqual(result.success_probability.b, Fraction(1, 8))
        self.assertGreater(len(updates), 2)
        self.assertEqual(updates[0].phase, "searching")
        self.assertEqual(updates[-1].phase, "complete")
        self.assertTrue(all(event.total is None for event in updates))
        counts = [event.solved_projectors for event in updates]
        self.assertEqual(counts, sorted(counts))
        bounds = [event.best_probability for event in updates]
        self.assertEqual(bounds, sorted(bounds))
        self.assertAlmostEqual(bounds[-1], result.probability)
        self.assertEqual(counts[-1], result.solved_projectors)

    def test_budget_failure_never_reports_completion(self):
        updates = []
        solver = StabilizerDiscriminationSolver(e8_sign(), max_projectors=10,
                                               progress=updates.append)
        with self.assertRaises(SolverLimitError):
            solver.solve()
        self.assertEqual(updates[-1].phase, "failed")
        self.assertNotIn("complete", [event.phase for event in updates])

    def test_cached_policy_read_does_not_restart_progress(self):
        updates = []
        solver = StabilizerDiscriminationSolver(twoqubit_mixture(), record_policy=True,
                                               progress=updates.append)
        solver.solve()
        count = len(updates)
        solver.policy_tree()
        self.assertEqual(len(updates), count)

    def test_progress_options_are_validated(self):
        with self.assertRaises(TypeError):
            StabilizerDiscriminationSolver(e8_sign(), progress=42)
        for interval in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                StabilizerDiscriminationSolver(e8_sign(), progress_interval=interval)


if __name__ == "__main__":
    unittest.main()
